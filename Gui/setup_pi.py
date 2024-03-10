#!/usr/bin/python
'''
setup_pi.py
To enable the robot car, all pins need to be set to input mode before
motor driver is connected.
'''

import RPi.GPIO as GPIO
GPIO.cleanup()
