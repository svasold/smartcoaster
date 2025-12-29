#! /usr/bin/python2

from fileinput import filename
from this import d
import time
import sys
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

# LED ring configuration:
LED_COUNT = 24        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
####!!dev-note: 20 scheint gut zu sein unter der Box. Zuviel Licht für zu Spiegelungen!
LED_BRIGHTNESS = 20  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

#Reboot the spectrometer, just in case
spec1.soft_reset()
spec4.soft_reset()

#Set the gain of the device between 0 and 3.  Higher gain = higher readings
spec1.set_gain(3)
spec4.set_gain(3)

#Set the integration time between 1 and 255.  Higher means longer readings
spec1.set_integration_time(50)
spec4.set_integration_time(50)

#Set the board to continuously measure all colours
spec1.set_measurement_mode(2)
spec4.set_measurement_mode(2)

#set led current (0 = lowest)
spec1.set_led_current(0)
spec4.set_led_current(0)

EMULATE_HX711 = False

referenceUnit = 440

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711


def cleanAndExit():
    print("Cleaning...")
    if not EMULATE_HX711:
        GPIO.cleanup()
    print("Bye!")
    sys.exit()

def printStatus():
    lcd.lcd_clear()

    lcd.lcd_display_string("user: " + users[user_index], 1)
    lcd.lcd_display_string("drink: " + drinks[drink_index], 2)
    lcd.lcd_display_string("--> photo", 3)

def getHighestImageIndex():
    onlyfiles = [(f.split("_"))[0] for f in listdir(data_path) if isfile(join(data_path, f))]
    onlyfiles = sorted(onlyfiles)
    if len(onlyfiles) == 0:
        return 0
    return int(os.path.splitext(onlyfiles[-1])[0])

hx = HX711(29, 31)
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # first button
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # second button
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#set pin for camera switch
cam_switch = 36
#GPIO.setmode(GPIO.BCM)
GPIO.setup(cam_switch,GPIO.OUT)

#init temperature sensor
temp_sensor = MLX90614()

# I've found out that, for some reason, the order of the bytes is not always the same between versions of python, numpy and the hx711 itself.
# Still need to figure out why does it change.
# If you're experiencing super random values, change these values to MSB or LSB until to get more stable values.
# There is some code below to debug and log the order of the bits and the bytes.
# The first parameter is the order in which the bytes are used to build the "long" value.
# The second paramter is the order of the bits inside each byte.
# According to the HX711 Datasheet, the second parameter is MSB so you shouldn't need to modify it.
#hx.set_reading_format("MSB", "MSB")


# HOW TO CALCULATE THE REFFERENCE UNIT
# To set the reference unit to 1. Put 1kg on your sensor or anything you have and know exactly how much it weights.
# In this case, 92 is 1 gram because, with 1 as a reference unit I got numbers near 0 without any weight
# and I got numbers around 184000 when I added 2kg. So, according to the rule of thirds:
# If 2000 grams is 184000 then 1000 grams is 184000 / 2000 = 92.
# hx.set_reference_unit(113)
hx.set_reference_unit(referenceUnit)
hx.reset()
hx.tare()

lcd = lcddriver.lcd()

camera = PiCamera()

data_path = '/home/pi/git_repo/smartcoaster/images/'

drinks = ['water', 'beer', 'wine', 'coffee']
drink_index = 0

users = ['marco', 'simon', 'max', 'hannes', 'kai', 'eduardo', 'granit']
user_index = 0

printStatus()

# Create NeoPixel object with appropriate configuration.
ring = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
ring.begin()
while True:
    try:
        if GPIO.input(11) == GPIO.LOW and GPIO.input(13) == GPIO.LOW and GPIO.input(15) == GPIO.LOW:
            cleanAndExit()
        if GPIO.input(11) == GPIO.LOW:  # next user
            user_index += 1
            if user_index >= len(users):
                user_index = 0
            printStatus()
            continue
        if GPIO.input(13) == GPIO.LOW: # next drink
            drink_index += 1
            if drink_index >= len(drinks):
                drink_index = 0
            printStatus()
            continue
        if GPIO.input(15) == GPIO.LOW or keyboard.is_pressed('p'): # take picture and store metadata
            lcd.lcd_clear()
            lcd.lcd_display_string('Smile...', 1)
            
            for i in range(0, ring.numPixels(), 1):
                ring.setPixelColor(i, Color(255,255,255))
            ring.show()
            #take images and switch cams

            GPIO.output(cam_switch,GPIO.LOW)
            file_name1 = str(getHighestImageIndex() + 1) + '_a.jpg'
            file_name2 = str(getHighestImageIndex() + 1) + '_b.jpg'
            camera.capture(data_path + "tmp_a.jpg")
            GPIO.output(cam_switch,GPIO.HIGH)
            time.sleep(1)
            camera.capture(data_path + "tmp_b.jpg")
            for i in range(0, ring.numPixels(), 1):
                ring.setPixelColor(i, Color(0,0,0))
            ring.show()
            weight = hx.get_weight(5)
            
            #get spectro readings
            spec1.enable_main_led()
            spec1_results = [str(element) for element in spec1.get_calibrated_values()]
            time.sleep(1)
            spec1.disable_main_led()
            spec1_result_string = ";".join(spec1_results)

            spec4.enable_main_led()
            spec4_results = [str(element) for element in spec4.get_calibrated_values()]
            time.sleep(1)
            spec4.disable_main_led()
            spec4_result_string = ";".join(spec4_results)

            object_temperature = temp_sensor.readObjectTemperature()
            ambient_temperature = temp_sensor.readAmbientTemperature()

            meta_data = {
                'user': users[user_index],
                'drink': drinks[drink_index],
                'weight': weight,
                'object_temperature': object_temperature,
                'ambient_temperature': ambient_temperature,
                'spec_bottom' : spec1_result_string,
                'spec_left' : spec4_result_string
            }
            with open(data_path + "tmp_a.jpg", "rb") as img_file:
                img_a = Image(img_file)
            with open(data_path + "tmp_b.jpg", "rb") as img_file:
                img_b = Image(img_file)
            #img.image_description = json.dumps(meta_data)
            img_a.user_comment = json.dumps(meta_data)
            img_b.user_comment = json.dumps(meta_data)

            with open(data_path + file_name1, 'wb') as updated_img:
                updated_img.write(img_a.get_file())
            with open(data_path + file_name2, 'wb') as updated_img:
                updated_img.write(img_b.get_file())

            os.remove(data_path + "tmp_a.jpg")
            os.remove(data_path + "tmp_b.jpg")

            
            lcd.lcd_display_string('pictures taken', 1)
            time.sleep(2)
            printStatus()

            
            



    except (KeyboardInterrupt, SystemExit):
            lcd.lcd_clear()
            #Set the board to measure just once (it stops after that)
            spec1.set_measurement_mode(3)
            spec4.set_measurement_mode(3)
            #Turn off the main LED
            spec1.disable_main_led()
            spec4.disable_main_led()

            for i in range(0, ring.numPixels(), 1):
                ring.setPixelColor(i, Color(0,0,0))
            ring.show()

            cleanAndExit()
