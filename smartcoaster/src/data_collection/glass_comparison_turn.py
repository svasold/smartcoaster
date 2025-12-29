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
LED_COUNT = 24  # Number of LED pixels.
LED_PIN = 18  # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10  # DMA channel to use for generating signal (try 10)
####!!dev-note: 20 scheint gut zu sein unter der Box. Zuviel Licht für zu Spiegelungen!
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0  # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Constants
ITERATIONS = 10

# Reboot the spectrometer, just in case
spec1.soft_reset()
spec4.soft_reset()

# Set the gain of the device between 0 and 3.  Higher gain = higher readings
spec1.set_gain(3)
spec4.set_gain(3)

# Set the integration time between 1 and 255.  Higher means longer readings
spec1.set_integration_time(50)
spec4.set_integration_time(50)

# Set the board to continuously measure all colours
spec1.set_measurement_mode(2)
spec4.set_measurement_mode(2)

# set led current (0 = lowest)
spec1.set_led_current(0)
spec4.set_led_current(0)


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

EMULATE_HX711 = False

referenceUnit = 440

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711


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

# Create NeoPixel object with appropriate configuration.
ring = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
ring.begin()

lcd = lcddriver.lcd()

# step for ring brightness
step = 256 // 4
# levels for brightness (ring and spectro led)
# levels = [(0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (1 * step, 3), (2 * step, 3), (3 * step, 3), (4 * step - 1, 3)]
LEVEL = 3
spec1.set_led_current(LEVEL)
spec4.set_led_current(LEVEL)

printLcd("ready")
while True:
    try:
        # 3 button exit
        if GPIO.input(11) == GPIO.LOW and GPIO.input(13) == GPIO.LOW and GPIO.input(15) == GPIO.LOW:
            cleanAndExit()
        # button 1 start measuring
        if GPIO.input(11) == GPIO.LOW:  # measure with different current levels
            printLcd("starting")
            # lists of spectra means
            spec1_mean = []
            spec4_mean = []
            #lists of stds (iterations)
            spec1_std = []
            spec4_std = []
            # different lighting conditions
            while True:
                if GPIO.input(13) == GPIO.LOW:  # measure with different current levels
                    # lists with spectra channel values (for each iteration)
                    spec1_results = []
                    spec4_results = []
                    printLcd("starting measurement")
                    spec1.enable_main_led()
                    time.sleep(1)
                    for i in range(ITERATIONS):
                        printLcd("1: run " + str(i))
                        # append the current spectra values
                        spec1_results.append(spec1.get_calibrated_values())
                        time.sleep(0.1)
                    spec1.disable_main_led()
                    spec1_results_np = numpy.array(spec1_results)
                    # append the relative standard deviation and mean
                    spec1_mean.append(numpy.mean(spec1_results_np, axis=0))
                    spec1_std.append(numpy.std(spec1_results_np, axis=0) / numpy.mean(spec1_results_np, axis=0))
                    print("spec1:", spec1_results_np)
                    spec4.enable_main_led()
                    time.sleep(1)
                    for i in range(ITERATIONS):
                        printLcd("4: run " + str(i))
                        # append the current spectra values
                        spec4_results.append(spec4.get_calibrated_values())
                        time.sleep(0.1)
                    spec4.disable_main_led()
                    spec4_results_np = numpy.array(spec4_results)
                    # append the relative standard deviation and mean
                    spec4_mean.append(numpy.mean(spec4_results_np, axis=0))
                    spec4_std.append(numpy.std(spec4_results_np, axis=0) / numpy.mean(spec4_results_np, axis=0))
                    print("spec4:", spec4_results_np)
                    printLcd("replace glass")
                    print("---------------------------------------------------------------------------")
                if GPIO.input(15) == GPIO.LOW:
                    printLcd("writing to file")
                    spec1_mean_mean = numpy.mean(numpy.array(spec1_mean), axis=0)
                    spec4_mean_mean = numpy.mean(numpy.array(spec4_mean), axis=0)
                    spec1_mean_std = numpy.std(numpy.array(spec1_mean), axis=0) / spec1_mean_mean
                    spec4_mean_std = numpy.std(numpy.array(spec4_mean), axis=0) / spec4_mean_mean

                    # writing deviations into file
                    with open('glass_comparison_turn.txt', 'w') as f:
                        writeMatrixToFile(f, "spec1 mean values over " + str(ITERATIONS) + " iterations (same postition):", spec1_mean)
                        writeMatrixToFile(f, "spec4 mean values over " + str(ITERATIONS) + " iterations (same postition):", spec4_mean)
                        writeVectorToFile(f, "spec1 std of mean values over different positions", spec1_mean_std)
                        writeVectorToFile(f, "spec4 std of mean values over different positions", spec4_mean_std)
                        writeVectorToFile(f, "spec1 mean over different positions", spec1_mean_mean)
                        writeVectorToFile(f, "spec4 mean over different positions", spec4_mean_mean)
                        writeMatrixToFile(f, "spec1 std over " + str(ITERATIONS) + " iterations (same postition):", spec1_std)
                        writeMatrixToFile(f, "spec4 std over " + str(ITERATIONS) + " iterations (same postition):", spec4_std)

                        f.close()
                    break

            printLcd("ready")

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
