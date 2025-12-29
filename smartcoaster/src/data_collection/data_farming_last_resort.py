#! /usr/bin/python2


from fileinput import filename
from this import d
import time
import sys

import numpy
import numpy as np

import lcddriver
import RPi.GPIO as GPIO
import xml.etree.ElementTree as ET
import MFRC522
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

# LED ring configuration:
LED_COUNT = 24  # Number of LED pixels.
LED_PIN = 18  # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10  # DMA channel to use for generating signal (try 10)
####!!dev-note: 20 scheint gut zu sein unter der Box. Zuviel Licht fuer zu Spiegelungen!
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0  # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Constants
ITERATIONS = 5

# camera image folder
data_path = '/home/pi/git_repo/smartcoaster/images_dark_onlyglass/'

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


# set led current (0 = lowest)
# spec1.set_led_current(0)
# spec4.set_led_current(0)


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
        # for i in range(1, 6):
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
    lcd.lcd_display_string("drink: " + drink, 2)
    lcd.lcd_display_string("options", 3)


def lcdOptions():
    lcd.lcd_clear()
    lcd.lcd_display_string("print csv", 1)
    lcd.lcd_display_string("delete last row", 2)
    lcd.lcd_display_string("back", 3)


def getHighestImageIndex():
    onlyfiles = [int(f[:-4]) for f in listdir(data_path) if isfile(join(data_path, f))]
    onlyfiles = sorted(onlyfiles)
    # print(onlyfiles)
    if len(onlyfiles) == 0:
        return 0
    return onlyfiles[-1]


EMULATE_HX711 = False
referenceUnit = 453

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711

hx = HX711(29, 31)
# button init
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # first button
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # second button
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)

lcd = lcddriver.lcd()

camera = PiCamera()

lcdUpdate("remove glass and press button 1")
# HOW TO CALCULATE THE REFFERENCE UNIT
# To set the reference unit to 1. Put 1kg on your sensor or anything you have and know exactly how much it weights.
# In this case, 92 is 1 gram because, with 1 as a reference unit I got numbers near 0 without any weight
# and I got numbers around 184000 when I added 2kg. So, according to the rule of thirds:
# If 2000 grams is 184000 then 1000 grams is 184000 / 2000 = 92.
# hx.set_reference_unit(113)
while True:
    if GPIO.input(11) == GPIO.LOW:
        break
hx.set_reference_unit(referenceUnit)
hx.reset()
hx.tare()

# Create NeoPixel object with appropriate configuration.
ring = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
ring.begin()

LEVEL = 3
# step for ring brightness
step = 256 // 4
# levels for brightness (ring and spectro led)
# levels = [0, 1 * step, 2 * step, 3 * step, 4 * step - 1]
levels = [0]
drinks = ['water', 'beer', 'red wine', 'white wine', 'apple juice', 'cola', 'fanta', 'mineral water', 'wheat beer',
          'orange juice', 'empty glass', 'tea', 'coffee', 'milk coffee']
drink_index = 0
drink = 'water'
spec1.set_led_current(LEVEL)
spec4.set_led_current(LEVEL)
GLASS_WEIGHT = 313  # sagres
PATH = '/home/pi/git_repo/smartcoaster/smartcoaster/data_spec/'
CSV_FILE = PATH + 'data_dark_onlyglass.csv'
DETAIL_CSV_FILE = PATH + 'data_detail_final.csv'
container = "glass"

lcdStatus()
while True:
    try:
        # button 1 measuring and writing to csv file
        if GPIO.input(11) == GPIO.LOW:  # measure with different current levels
            # weight
            weight = round(hx.get_weight(5)) - GLASS_WEIGHT
            with open(CSV_FILE, 'a') as csvfile:
                writer = csv.writer(csvfile)  # this is the writer object
                # spec4_results = [[]]
                spec4_results = np.zeros((2, ITERATIONS, 6))
                for brightness_index, spec_led_brightness in enumerate([1, 3]):
                    lcdUpdate("led: " + str(spec_led_brightness))
                    spec4.set_led_current(spec_led_brightness)
                    spec4.enable_main_led()
                    time.sleep(2)
                    for i in range(ITERATIONS):
                        # from the top
                        # spec4_results[brightness_index].append(spec4.get_calibrated_values())
                        spec4_results[brightness_index, i] = spec4.get_calibrated_values()
                    time.sleep(2)
                    spec4.disable_main_led()

                for sample_index in range(ITERATIONS):
                    dataset = []
                    dataset.append(drink)
                    dataset.append(weight)
                    dataset.append(container)
                    dataset.append(0)
                    dataset.append("no pic")
                    dataset.extend(numpy.zeros(6))
                    dataset.extend(spec4_results[0][sample_index])
                    dataset.extend(numpy.zeros(6))
                    dataset.extend(spec4_results[1][sample_index])
                    writer.writerow(dataset)
            lcdStatus()

        # button 2
        if GPIO.input(13) == GPIO.LOW or keyboard.is_pressed(' '):
            drink_index += 1
            if drink_index >= len(drinks):
                drink_index = 0
            drink = drinks[drink_index]
            lcdStatus()
            time.sleep(0.5)
            continue

        # button 3 reading csv file and printing it
        if GPIO.input(15) == GPIO.LOW:
            lcdOptions()
            time.sleep(0.5)
            while True:
                if GPIO.input(11) == GPIO.LOW:
                    printCsv()
                    time.sleep(0.5)

                if GPIO.input(13) == GPIO.LOW:
                    deleteLastRow()
                    lcdOptions()
                if GPIO.input(15) == GPIO.LOW:
                    lcdStatus()
                    time.sleep(0.5)
                    break






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
