#! /usr/bin/python2


from fileinput import filename
import time
import sys

import numpy

import lcddriver
import RPi.GPIO as GPIO
import xml.etree.ElementTree as ET
import MFRC522
import signal
import datetime
import time
import os
from picamera import PiCamera
from exif import Image
from os import listdir
from os.path import isfile, join
import json
import AS7262_Pi_bus1 as spec1
import AS7262_Pi_bus4 as spec4
from MLX90614 import *
from rpi_ws281x import PixelStrip, Color
import argparse
import keyboard
import csv
import RPi.GPIO as GPIO
from hx711 import HX711

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn import svm

import requests

from requests.exceptions import HTTPError
from google.oauth2.credentials import Credentials
from google.cloud.firestore import Client

# daten von smartcoaster
nfcUid = "fe8abb7b"
drink = "water"
weight = 200
date = datetime.datetime.now()
isNewDrink = True
spectroData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
container = "mug"



########################## paths to data ##########################
# camera image folder
#data_path = '/home/pi/git_repo/smartcoaster/images_test_setup1/' #test data
data_path = '/home/pi/git_repo/smartcoaster/images_main/'     #training data

# csv file with measured values
PATH = '/home/pi/git_repo/smartcoaster/smartcoaster/data_spec/'
#CSV_FILE = PATH + 'test_data_setup1.csv' # test data
CSV_FILE = PATH + 'data_dark_onlyglass.csv'
###################################################################


SCALE_REFERENCE_UNIT = 453

GLASS_WEIGHT = 313  # sagres

# STEP_SIZE for ring brightness
STEP_SIZE = 256 // 4
# levels for brightness (ring and spectro led)
BRIGHTNESS_LEVELS = [0, 1 * STEP_SIZE, 2 * STEP_SIZE, 3 * STEP_SIZE, 4 * STEP_SIZE - 1]
DRINKS_LIST = ['water', 'beer', 'red wine', 'white wine', 'apple juice', 'cola', 'fanta', 'mineral water', 'wheat beer', 'orange juice', 'empty glass']
CONTAINER_LIST = ['glass', 'wide glass', 'mug']
CONTAINER_WEIGHT_LIST = [313, 835, 616]

STARTING_DRINK_INDEX = 0
STARTING_DRINK = DRINKS_LIST[STARTING_DRINK_INDEX]
STARTING_CONTAINER_INDEX = 0

drink_index = STARTING_DRINK_INDEX
drink = DRINKS_LIST[STARTING_DRINK_INDEX]
container_index = STARTING_CONTAINER_INDEX
container = CONTAINER_LIST[STARTING_CONTAINER_INDEX]
container_weight = CONTAINER_WEIGHT_LIST[STARTING_CONTAINER_INDEX]

current_uid = ""

drink_dic = {
    'apple juice' : 'applejuice',
    'beer' : 'beer',
    'cola' : 'coke',
    'fanta' : 'fanta',
    'mineral water' : 'mineral_water',
    'orange juice' : 'orange_juice',
    'red wine' : 'wine_red',
    'white wine' : 'wine_white',
    'water' : 'water',
    'coffee' : 'coffee_black',
    'milk coffee' : 'coffee_milk',
    'tea' : 'tee'
}


# Constants
MEASUREMENT_ITERATIONS = 5


# Create an object of the class MFRC522
MIFAREReader = MFRC522.MFRC522()

####################### scale configuration #######################
hx = HX711(29, 31)

# HOW TO CALCULATE THE REFFERENCE UNIT
# To set the reference unit to 1. Put 1kg on your sensor or anything you have and know exactly how much it weights.
# In this case, 92 is 1 gram because, with 1 as a reference unit I got numbers near 0 without any weight
# and I got numbers around 184000 when I added 2kg. So, according to the rule of thirds:
# If 2000 grams is 184000 then 1000 grams is 184000 / 2000 = 92.
# hx.set_reference_unit(113)
hx.set_reference_unit(SCALE_REFERENCE_UNIT)
###################################################################

###################### LED ring configuration ######################
LED_COUNT = 24        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
####!!dev-note: 20 scheint gut zu sein unter der Box. Zuviel Licht fuer zu Spiegelungen!
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Create NeoPixel object with appropriate configuration.
ring = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
ring.begin()
###################################################################

################ Spectroscopy sensor configuration ################
# Reboot the spectrometer, just in case
spec1.soft_reset()
spec4.soft_reset()

# Set the gain of the device between 0 and 3.  Higher gain = higher readings
spec1.set_gain(3)
spec4.set_gain(3)

# Set the integration time between 1 and 255.  Higher means longer readings
spec1.set_integration_time(70)
spec4.set_integration_time(70)

# Set the board to continuously measure all colours
spec1.set_measurement_mode(2)
spec4.set_measurement_mode(2)
###################################################################

####################### button configuration ######################
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # first button
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # second button
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # third button
###################################################################


def writeMatrixToFile(f, title, matrix):
    f.write(title + "\n")
    for row in matrix:
        for column in row:
            f.write("{:.4f}".format(column).ljust(10))
        f.write("\n")
    f.write("\n")


def writeVectorToFile(f, title, vector):
    f.write(title + "\n")
    for value in vector:
        f.write("{:.4f}".format(value).ljust(10))
    f.write("\n")
    f.write("\n")


def printList(list):
    for row in list:
        row_string = ""
        for column in row:
            row_string += str(column).ljust(20)
        print(row_string)
    print("\n")

def cleanAndExit():
    print("Cleaning...")
    GPIO.cleanup()
    print("Bye!")
    sys.exit()


def lcdUpdate(string):
    lcd.lcd_clear()
    lcd.lcd_display_string(string, 1)

def deleteLastRow():
    lcd.lcd_clear()
    lcd.lcd_display_string("yes", 1)
    lcd.lcd_display_string("no", 2)
    lcd.lcd_display_string("delete row?", 3)
    time.sleep(0.5)
    while True:
        if GPIO.input(11) == GPIO.LOW:
            break
        if GPIO.input(13) == GPIO.LOW:
            return

    lcdUpdate("deleting row...")
    os.rename(CSV_FILE, PATH + 'tmp.csv')
    with open(PATH + 'tmp.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        file_content = list(reader)
        #for i in range(1, 6):
        file_content.pop()

    with open(CSV_FILE, 'w') as csvfile:
        # numpy.savetxt(csvfile, spec4_median, delimiter=',')
        writer = csv.writer(csvfile)  # this is the writer object
        writer.writerows(file_content)  # this is the data
    os.remove(PATH + 'tmp.csv')
    time.sleep(1)

def printCsv():
    with open(CSV_FILE, 'r') as csvfile:
        # data = numpy.loadtxt(csvfile, delimiter=',')
        reader = csv.reader(csvfile)
        data = list(reader)
        printList(data)
        # print(data.shape)
        print("\n")

def lcdStatus():
    lcd.lcd_clear()
    lcd.lcd_display_string("start measurement", 1)
    lcd.lcd_display_string("edit container/drink", 2)
    lcd.lcd_display_string("options", 3)

def lcdOptions3():
    lcd.lcd_clear()
    lcd.lcd_display_string("log out", 1)
    lcd.lcd_display_string("delete last row", 2)
    lcd.lcd_display_string("back", 3)

def lcdOptions2():
    lcd.lcd_clear()
    lcd.lcd_display_string("drink: " + drink, 1)
    lcd.lcd_display_string("container: " + container, 2)
    lcd.lcd_display_string("back", 3)

def lcdOptions1():
    lcd.lcd_clear()
    lcd.lcd_display_string("collect Data", 1)
    lcd.lcd_display_string("classify Data", 2)
    lcd.lcd_display_string("back", 3)

def lcdStart():
    lcd.lcd_clear()
    lcd.lcd_display_string("log in", 1)
    lcd.lcd_display_string("get RFID UID", 2)
    lcd.lcd_display_string("shut down", 3)

def getHighestImageIndex():
    onlyfiles = [int(f[:-4]) for f in listdir(data_path) if isfile(join(data_path, f))]
    onlyfiles = sorted(onlyfiles)
    #print(onlyfiles)
    if len(onlyfiles) == 0:
        return 0
    return onlyfiles[-1]

def nextContainer():
    global container_index, CONTAINER_LIST, container, container_weight

    container_index += 1
    if container_index >= len(CONTAINER_LIST):
        container_index = 0
    container = CONTAINER_LIST[container_index]
    container_weight = CONTAINER_WEIGHT_LIST[container_index]

def nextDrink():
    global drink_index, DRINKS_LIST, drink

    drink_index += 1
    if drink_index >= len(DRINKS_LIST):
        drink_index = 0
    drink = DRINKS_LIST[drink_index]

def takeAndSavePicture():
    if(container != "mug"):
        #take camera pic
        file_name = str(getHighestImageIndex() + 1) + '.jpg'
        #print(getHighestImageIndex())
        camera.capture(data_path + "tmp.jpg")
        with open(data_path + "tmp.jpg", "rb") as img_file:
            img = Image(img_file)
        with open(data_path + file_name, 'wb') as updated_img:
            updated_img.write(img.get_file())
        os.remove(data_path + "tmp.jpg")
    else:
        file_name = "no-file"
    return file_name

def setRing(level):
    global ring
    for i in range(0, ring.numPixels(), 1):
        ring.setPixelColor(i, Color(level, level, level))
    ring.show()

def takeSpectroMeasurement(spectro_nr):
    ret_val = []
    if spectro_nr == 1:
        spec1.enable_main_led()
        ret_val = spec1.get_calibrated_values()
        spec1.disable_main_led()
    elif spectro_nr == 4:
        spec4.enable_main_led()
        ret_val = spec4.get_calibrated_values()
        spec4.disable_main_led()
    return ret_val

# Use the google verify password REST api to authenticate and generate user tokens
def sign_in_with_email_and_password(api_key, email, password):
    FIREBASE_REST_API = "https://identitytoolkit.googleapis.com/v1/accounts"
    request_url = "%s:signInWithPassword?key=%s" % (FIREBASE_REST_API, api_key)
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"email": email, "password": password, "returnSecureToken": True})
    
    resp = requests.post(request_url, headers=headers, data=data)
    # Check for errors
    try:
        resp.raise_for_status()
    except HTTPError as e:
        raise HTTPError(e, resp.text)
        
    return resp.json()

def handleDbCommuication(nfcUid, drink, weight, date, isNewDrink, spectroData, container):
    # Credentials are loaded from environment for publication safety. Set these before running.
    apiKey = os.environ.get("SMARTCOASTER_FIREBASE_API_KEY", "<set-api-key>")
    email = os.environ.get("SMARTCOASTER_FIREBASE_EMAIL", "<set-email>")
    password = os.environ.get("SMARTCOASTER_FIREBASE_PASSWORD", "<set-password>")
    projectId = os.environ.get("SMARTCOASTER_FIREBASE_PROJECT", "<set-project>")
    
    # We use the sign_in_with_email_and_password function from https://gist.github.com/Bob-Thomas/49fcd13bbd890ba9031cc76be46ce446
    response = sign_in_with_email_and_password(apiKey, email, password)

    # Use google.oauth2.credentials and the response object to create the correct user credentials
    creds = Credentials(response['idToken'], response['refreshToken'])

    # Use the raw firestore grpc client instead of building one through firebase_admin
    db = Client(projectId, creds)

    try:
        # get userUid 
        nfcDoc = db.collection('nfcCards').document(nfcUid).get()
        userUid = nfcDoc.get('userUid')
        
        # add drink 
        drinkObject = {
            "userUid": userUid,
            "drink": drink,
            "weight": weight,
            "date": date,
            "newDrink": isNewDrink
        }
        response = db.collection("drinks").add(drinkObject)
        drinkDocId = response[1].id
        
        # add model data
        modelData = {
            "drink": drink,
            "weight": weight,
            "container": container,
            "drinkDocId": drinkDocId,
            "spectroData": spectroData
        }
        response = db.collection("modelData").add(modelData)

    except: 
        print("error: something went wrong!")



def collectData():
    # weight
    weight = round(hx.get_weight(5)) - container_weight

    

    file_name = takeAndSavePicture()
    dataset = []
    for i in range(MEASUREMENT_ITERATIONS):
        dataset.append([])
        dataset[i].append(drink)
        dataset[i].append(weight)
        dataset[i].append(container)
        dataset[i].append(0)
        dataset[i].append(file_name)
        for spec_led_brightness in [1, 3]:
            lcdUpdate(" led: " + str(spec_led_brightness))
            spec1.set_led_current(spec_led_brightness)
            spec4.set_led_current(spec_led_brightness)

            #from the side
            if(container != "mug"):
                dataset[i].extend(takeSpectroMeasurement(1))
            else:
                dataset[i].extend([0, 0, 0, 0, 0, 0])

            #from the top
            dataset[i].extend(takeSpectroMeasurement(4))
    # deactivate led ring
    setRing(0)
    return dataset

def collectDataSlow():
    # weight
    weight = round(hx.get_weight(5)) - container_weight
    spec4_results = numpy.zeros((2, MEASUREMENT_ITERATIONS, 6))
    for brightness_index, spec_led_brightness in enumerate([1, 3]):
        lcdUpdate("led: " + str(spec_led_brightness))
        spec4.set_led_current(spec_led_brightness)
        spec4.enable_main_led()
        time.sleep(2)
        for i in range(MEASUREMENT_ITERATIONS):
            spec4_results[brightness_index, i] = spec4.get_calibrated_values()
        time.sleep(2)
        spec4.disable_main_led()

    dataset = []
    for sample_index in range(MEASUREMENT_ITERATIONS):
        dataset.append([])
        dataset[sample_index].append(drink)
        dataset[sample_index].append(weight)
        dataset[sample_index].append(container)
        dataset[sample_index].append(0)
        dataset[sample_index].append("no pic")
        dataset[sample_index].extend(numpy.zeros(6))
        dataset[sample_index].extend(spec4_results[0][sample_index])
        dataset[sample_index].extend(numpy.zeros(6))
        dataset[sample_index].extend(spec4_results[1][sample_index])


    return dataset

def lcdPrediction(prediction):
    lcd.lcd_clear()
    lcd.lcd_display_string("prediction:", 1)
    lcd.lcd_display_string("-> " + prediction, 2)
    lcd.lcd_display_string("back", 3)
def lcdClassifyWait():
    lcd.lcd_clear()
    lcd.lcd_display_string("Classifying...", 1)


def getSamples(csv_data, csv_columns, min_max_scaler, scaling = True, container = None):
    if container:
        data = csv_data[csv_data[:, 2] == container]
    else:
        data = csv_data
    # discard every fifth sample
    indices = numpy.arange(len(data))
    data = data[indices % 5 != 0]
    features = data[:, csv_columns[0:-1]]
    targets = data[:, csv_columns[-1]]

    if scaling:
        features = min_max_scaler.fit_transform(features)

    return features, targets

def classifyData():
    samples = collectDataSlow()#collectData()
    lcdClassifyWait()
    samples = numpy.asarray(samples)
    print(samples)
    spec_values = samples[:,samples.shape[1]-24:samples.shape[1]].astype(numpy.float)
    print(spec_values)
    median_spec_values = numpy.median(spec_values, axis=0)
    assert(len(median_spec_values)==24)
    print(median_spec_values)
    classify_sample = numpy.append(median_spec_values, samples[0][1])
    print(classify_sample)


    # Read Data
    with open(CSV_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        csv_data = numpy.array(list(reader))
    # global variabels
    #csv_columns = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 1, 0]
    csv_columns = list(range(5, csv_data.shape[1]))
    csv_columns.extend([1, 0])
    min_max_scaler = preprocessing.MinMaxScaler()
    training_samples, training_targets = getSamples(csv_data, csv_columns, min_max_scaler, container="glass")
    # assert (training_samples.shape[1] == 13)
    lda = LinearDiscriminantAnalysis(n_components=10)
    training_samples = lda.fit_transform(training_samples, training_targets)
    model = svm.SVC(kernel='poly', class_weight='balanced', degree=1)
    # model = LinearDiscriminantAnalysis()
    # model = KNeighborsClassifier(n_neighbors=1)
    model.fit(training_samples, training_targets)

    classify_sample = min_max_scaler.transform(classify_sample.reshape((1, -1)))
    classify_sample = lda.transform(classify_sample)
    prediction = model.predict(classify_sample)
    print(prediction)
    date = datetime.datetime.now()
    handleDbCommuication(current_uid, drink_dic[prediction[0]], int(samples[0][1]), date, isNewDrink, median_spec_values.tolist(), container)

    lcdPrediction(prediction[0])
    while(1):
        if GPIO.input(15) == GPIO.LOW:
            lcdStatus()
            time.sleep(0.5)
            break


def run():
    log_out = False
    while True:
        setRing(255)
        # button 1 measuring and writing to csv file
        if GPIO.input(11) == GPIO.LOW:  # measure with different current levels
            lcdOptions1()
            time.sleep(0.5)
            while True:
                if GPIO.input(11) == GPIO.LOW:
                    with open(CSV_FILE, 'a') as csvfile:
                        writer = csv.writer(csvfile)  # this is the writer object
                    samples = collectData()
                    for row in samples:
                        writer.writerow(row)
                    lcdStatus()
                    time.sleep(0.5)
                    break
                if GPIO.input(13) == GPIO.LOW:
                    classifyData()
                    lcdStatus()
                    time.sleep(0.5)
                    break
                if GPIO.input(15) == GPIO.LOW:
                    lcdStatus()
                    time.sleep(0.5)
                    break

            

        # button 2
        if GPIO.input(13) == GPIO.LOW or keyboard.is_pressed(' '):
            lcdOptions2()
            time.sleep(0.5)
            while True:
                if GPIO.input(11) == GPIO.LOW:
                    nextDrink()
                    lcdOptions2()
                    time.sleep(0.5)
                    continue
                if GPIO.input(13) == GPIO.LOW:
                    nextContainer()
                    lcdOptions2()
                    time.sleep(0.5)
                    continue
                if GPIO.input(15) == GPIO.LOW:
                    lcdStatus()
                    time.sleep(0.5)
                    break
            

        # button 3 reading csv file and printing it
        if GPIO.input(15) == GPIO.LOW:
            lcdOptions3()
            time.sleep(0.5)
            while True:
                if GPIO.input(11) == GPIO.LOW:
                    log_out = True
                    current_uid = ""
                    time.sleep(0.5)
                    break
                if GPIO.input(13) == GPIO.LOW:
                    lcdOptions2()
                    time.sleep(0.5)
                    while True:
                        if GPIO.input(11) == GPIO.LOW:
                            nextDrink()
                            lcdOptions2()
                            time.sleep(0.5)
                            continue
                        if GPIO.input(13) == GPIO.LOW:
                            nextContainer()
                            lcdOptions2()
                            time.sleep(0.5)
                            continue
                        if GPIO.input(15) == GPIO.LOW:
                            lcdOptions3()
                            time.sleep(0.5)
                            break
                if GPIO.input(15) == GPIO.LOW:
                    lcdStatus()
                    time.sleep(0.5)
                    break

        if log_out:
            lcdStart()
            break






def lcdUID(uid):
    lcd.lcd_clear()
    lcd.lcd_display_string("uid:", 1)
    lcd.lcd_display_string(str(uid), 2)
    lcd.lcd_display_string("back", 3)
def lcdUidSearch():
    lcd.lcd_clear()
    lcd.lcd_display_string("looking for card...", 1)
    lcd.lcd_display_string("back", 3)
def lcdTare():
    lcd.lcd_clear()
    lcd.lcd_display_string("remove container &", 1)
    lcd.lcd_display_string("press button 1", 3)



def showUID():
    foundCard = False
    lcdUidSearch()
    while(1):
        # Scan for cards    
        (status,TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)

        # If a card is found
        if status == MIFAREReader.MI_OK:
            print("found")

        (status,uid) = MIFAREReader.MFRC522_Anticoll()

        # If we have the UID, continue
        if status == MIFAREReader.MI_OK and not foundCard:
            uid_hex = ""
            for byte in uid:
                uid_hex += str(hex(byte))[2:]
            lcdUID(uid_hex)
            print(uid_hex)
            foundCard=True
        if GPIO.input(15) == GPIO.LOW:
            lcdStart()
            time.sleep(0.5)
            break
def logIn():
    global current_uid
    lcdUidSearch()
    while(1):
    # Scan for cards    
        (status,TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)

        # If a card is found
        if status == MIFAREReader.MI_OK:
            print("found")

        (status,uid) = MIFAREReader.MFRC522_Anticoll()

        # If we have the UID, continue
        if status == MIFAREReader.MI_OK:
            uid_hex = ""
            for byte in uid:
                uid_hex += str(hex(byte))[2:]
                if len(uid_hex) == 8:
                    break
            current_uid = uid_hex
            print(current_uid)
            lcdStatus()
            break
        if GPIO.input(15) == GPIO.LOW:
            lcdStart()
            time.sleep(0.5)
            break

    


lcd = lcddriver.lcd()

camera = PiCamera()

lcdTare()
while True:
    if GPIO.input(11) == GPIO.LOW:
        break
hx.reset()
hx.tare()




lcdStatus()
do_collect = False


try:
    lcdStart()
    while True:
        if GPIO.input(11) == GPIO.LOW:
            logIn()
            run()
            time.sleep(0.5)
            continue
        if GPIO.input(13) == GPIO.LOW:
            showUID()
            time.sleep(0.5)
            continue
        if GPIO.input(15) == GPIO.LOW:
            lcd.lcd_clear()
            lcd.lcd_display_string("Shutting down...", 1)
            os.system("sudo shutdown -h now")
except (KeyboardInterrupt, SystemExit):
    lcd.lcd_clear()
    # Set the board to measure just once (it stops after that)
    # spec1.set_measurement_mode(3)
    spec4.set_measurement_mode(3)
    # Turn off the main LED
    # spec1.disable_main_led()
    spec4.disable_main_led()

    # deactivate led ring
    for i in range(0, ring.numPixels(), 1):
        ring.setPixelColor(i, Color(0, 0, 0))
    ring.show()

    cleanAndExit()
