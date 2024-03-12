#!/usr/bin/python
'''
setup_pi.py
To enable the robot car, all pins need to be set to input mode before
motor driver is connected.
'''

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(7,GPIO.OUT)
GPIO.setup(8,GPIO.OUT)
GPIO.setup(9,GPIO.OUT)
GPIO.setup(10,GPIO.OUT)
GPIO.output(7,0)
GPIO.output(8,0)
GPIO.output(9,0)
GPIO.output(10,0)
GPIO.cleanup()