#! /usr/bin/python2

from fileinput import filename
from this import d
import time
import sys

import numpy

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
LED_COUNT = 24        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
####!!dev-note: 20 scheint gut zu sein unter der Box. Zuviel Licht für zu Spiegelungen!
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Constants
ITERATIONS = 5

EMULATE_HX711 = False

referenceUnit = 440

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711



camera = PiCamera()

data_path = '/home/pi/git_repo/smartcoaster/smartcoaster/camera_brightness_comparison_images/'

def cleanAndExit():
    print("Cleaning...")
    GPIO.cleanup()
    print("Bye!")
    sys.exit()


def printLcd(string):
    lcd.lcd_clear()
    lcd.lcd_display_string(string, 1)


hx = HX711(29, 31)
# button init
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # first button
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # second button
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#set pin for camera switch
cam_switch = 36
#GPIO.setmode(GPIO.BCM)
GPIO.setup(cam_switch,GPIO.OUT)

# Create NeoPixel object with appropriate configuration.
ring = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
ring.begin()

lcd = lcddriver.lcd()

# step for ring brightness
step = 256 // 8
# levels for brightness (ring and spectro led)
levels = [1*step, 2*step, 3*step, 4*step, 5*step, 6*step, 7*step, 8*step-1]

printLcd("ready")
while True:
    try:
        # 3 button exit
        if GPIO.input(11) == GPIO.LOW and GPIO.input(13) == GPIO.LOW and GPIO.input(15) == GPIO.LOW:
            cleanAndExit()
        # button 1 start measuring
        if GPIO.input(11) == GPIO.LOW:  # measure with different current levels
            printLcd("taking pics")
            # different lighting conditions
            for current_level in levels:
                # activate led ring
                for i in range(0, ring.numPixels(), 1):
                    ring.setPixelColor(i, Color(current_level, current_level, current_level))
                ring.show()
                
                for i in range(ITERATIONS):
                    #take images and switch cams
                    GPIO.output(cam_switch,GPIO.LOW)
                    camera.capture(data_path + "low_" + str(current_level) + "_" + str(i) + ".jpg")
                    GPIO.output(cam_switch,GPIO.HIGH)
                    time.sleep(1)
                    camera.capture(data_path + "high_" + str(current_level) + "_" + str(i) + ".jpg")


            # deactivate led ring
            for i in range(0, ring.numPixels(), 1):
                ring.setPixelColor(i, Color(0, 0, 0))
            ring.show()

            printLcd("ready")
            continue
        # button 2
        if GPIO.input(13) == GPIO.LOW:
            continue
        # button 3
        if GPIO.input(15) == GPIO.LOW or keyboard.is_pressed(' '):
            continue





    except (KeyboardInterrupt, SystemExit):
        lcd.lcd_clear()
        # Set the board to measure just once (it stops after that)
        spec1.set_measurement_mode(3)
        spec4.set_measurement_mode(3)
        # Turn off the main LED
        spec1.disable_main_led()
        spec4.disable_main_led()

        for i in range(0, ring.numPixels(), 1):
            ring.setPixelColor(i, Color(0, 0, 0))
        ring.show()

        cleanAndExit()
