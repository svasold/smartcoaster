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

EMULATE_HX711 = False

referenceUnit = 455

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
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    onlyfiles = sorted(onlyfiles)
    if len(onlyfiles) == 0:
        return 0
    return int(os.path.splitext(onlyfiles[-1])[0])

hx = HX711(29, 31)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # first button
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # second button
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)



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
#hx.set_reference_unit(referenceUnit)


lcd = lcddriver.lcd()

camera = PiCamera()

data_path = '/home/pi/smartcoaster/images/'

drinks = ['water', 'beer', 'wine', 'coffee']
drink_index = 0

users = ['marco', 'simon', 'max', 'hannes', 'kai', 'eduardo', 'granit']
user_index = 0

printStatus()

while True:
    try:
        if GPIO.input(16) == GPIO.LOW and GPIO.input(18) == GPIO.LOW and GPIO.input(15) == GPIO.LOW:
            cleanAndExit()
        if GPIO.input(16) == GPIO.LOW:  # next user
            user_index += 1
            if user_index >= len(users):
                user_index = 0
            printStatus()
            continue
        if GPIO.input(18) == GPIO.LOW: # next drink
            drink_index += 1
            if drink_index >= len(drinks):
                drink_index = 0
            printStatus()
            continue
        if GPIO.input(15) == GPIO.LOW: # take picture and store metadata
            lcd.lcd_clear()
            lcd.lcd_display_string('Smile...', 1)
            
            file_name = str(getHighestImageIndex() + 1) + '.jpg'
            camera.capture(data_path + "tmp.jpg")
            weight = hx.get_weight(5)

            meta_data = {
                'user': users[user_index],
                'drink': drinks[drink_index],
                'weight': weight,
                'temperature': 0
            }
            with open(data_path + "tmp.jpg", "rb") as img_file:
                img = Image(img_file)
            #img.image_description = json.dumps(meta_data)
            img.user_comment = json.dumps(meta_data)

            with open(data_path + file_name, 'wb') as updated_img:
                updated_img.write(img.get_file())

            os.remove(data_path + "tmp.jpg")

            
            lcd.lcd_display_string('picture taken', 1)
            time.sleep(2)
            printStatus()

            
            



    except (KeyboardInterrupt, SystemExit):
                lcd.lcd_clear()
                cleanAndExit()