'''
utils.py

Utility functions for reading and writing GPIOs.
'''

import collections, time
import numpy as np
import matplotlib.pyplot as pl
import RPi.GPIO as GPIO 
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import threading

# GPIO stuff
def setup():
    '''Set up environment for GPIO'''
    GPIO.setmode(GPIO.BCM)
    return 0

def close():
    '''Close the environment and clean up'''
    GPIO.cleanup()
    return 0

# Classes for saving data in real time
class DynamicBuffer():
    '''Buffer class that stores data in real time for plotting purposes'''
    def __init__(self, T, dt, func=lambda: 0, init=0, trigger=None, outpin=None, relu=False, invert=False):
        '''
        Initialise buffer. The buffer is set to size=T/dt.
        Parameters:
            T (float)       : Total time duration to buffer (s)
            dt (float)      : Sample interval (s)
            func            : Function that returns the value to update
            init (float)    : Initial value of array
            trigger (float) : The value at which to trigger an output
            outpin (int)    : If trigger is enabled, send a TTL out from this pin
        '''
        self.T, self.dt = T, dt
        self.init = init
        self._interval = int(dt*1000)
        self._bufsize = int(T/dt)
        self.databuffer = collections.deque([init]*self._bufsize, self._bufsize)
        self.setFunc(func)
        self.setTrigger(trigger, outpin)
        self.relu, self.invert = False, False

    def setTrigger(self, trigger, outpin):
        '''Set up output trigger'''
        self.trigger = trigger
        self.outpin = outpin

        if self.trigger is not None:
            assert self.outpin
            GPIO.setup(self.outpin, GPIO.OUT)
        else: 
            if self.outpin is not None: GPIO.cleanup(self.outpin)

    def setFunc(self, func=lambda: 0):
        '''Set up function'''
        self.func = func

    def setParameters(self, T, dt):
        '''Apply new parameters to buffer'''
        self.T, self.dt = T, dt
        self._interval = int(dt*1000)
        self._bufsize = int(T/dt)
        self.databuffer = collections.deque([self.init]*self._bufsize, self._bufsize)

    def update(self):
        '''Update the buffer'''
        sample = self.func()
        # Check for relu
        if self.relu: sample = relu(sample)
        # Check for inversion 
        if self.invert: sample = 1-sample
        self.databuffer.append(sample)
        if self.trigger is not None: self._trigger(sample)

    def _trigger(self, sample):
        '''Trigger an output'''
        if sample > self.trigger: GPIO.output(self.outpin, 1)
        else: GPIO.output(self.outpin, 0)

class PinBuffer(DynamicBuffer):
    '''Data buffer for pin inputs. func will act on the input of the pin.'''
    def __init__(self, pin, T, dt, func=lambda x, *y: x, init=0, trigger=None, outpin=None):
        super().__init__(T, dt, init=init, trigger=trigger, outpin=outpin)
        self.pin = pin

        # Set up the pin
        GPIO.setup(self.pin, GPIO.IN)

        # Read pin input as function
        self.func = lambda: func(GPIO.input(self.pin), self.databuffer)

    def setFunc(self, func=lambda x, *y: x):
        '''Set up function'''
        self.func = lambda: func(GPIO.input(self.pin), self.databuffer)

class Integrator(DynamicBuffer):
    '''Integrate inputs'''
    def __init__(self, inputs, T, dt, func=lambda x, *y: x, init=0, trigger=None, outpin=None, delay=0, weights=1.):
        super().__init__(T, dt, init=init, trigger=trigger, outpin=outpin)
        self.inputs = inputs
        # Check weights
        self.setWeights(weights)

        # Check transmission delay
        self.setDelays(delay)

        # Make sure weights, delays, and inputs have same dimensions
        # assert len(self.inputs) == len(self.delay)
        # assert len(self.inputs) == len(self.weights)

        self.func = lambda: func(self._integrate(), self.databuffer)

    def setInputs(self, inputs):
        '''Set new inputs'''
        self.inputs = inputs

    def setWeights(self, weights):
        '''Set new weights'''
        if type(weights) is not list: self.weights = [weights/len(self.inputs)] * len(self.inputs)
        elif len(weights) < len(self.inputs): self.weights = weights + [0]*(len(self.inputs) - len(weights))
        else: self.weights = weights

    def setDelays(self, delays):
        '''Set new delays'''
        # Check transmission delay
        if type(delays) is not list: delays = [delays] * len(self.inputs)
        elif len(delays) < len(self.inputs): delays = delays + [0]*(len(self.inputs) - len(delays))
        self.delay = [int(d/self.dt)+1 for d in delays] # append one since we are counting backwards

    def _integrate(self):
        '''Sum over the last entries in inputs'''
        out = 0
        for i, input in enumerate(self.inputs): out += input.databuffer[-self.delay[i]] * self.weights[i]
        return out
    
    def setFunc(self, func=lambda x, *y: x):
        '''Set up function'''
        self.func = lambda: func(self._integrate(), self.databuffer)
    
# Function to augment databuffers   
def relu(x):
    '''Linear rectifying unit'''
    if x < 0: return 0
    elif x > 1: return 1
    else: return x

class expKernel():
    '''Apply an exponential kernel to simulated data mimicking neuron EPSPs'''
    def __init__(self, tau):
        self.tau = tau
        self.x = 0

    def __call__(self, x, *y):
        self.x = self.x + (x-self.x)/self.tau
        return min([self.x, 1])

class Sigmoid():
    '''Apply a sigmoid functoin to data'''
    
    def __call__(self, x, *y):
        return 1/(1+np.exp(-x))

class ReLU():
    '''Apply a sigmoid functoin to data'''
    
    def __call__(self, x, *y):
        if x < 0: return 0
        elif x > 1: return 1
        else: return x

if __name__ == '__main__':
    # Test PinBuffer
    setup()
    T, dt = 1, 0.001

    kernel = expKernel(tau=50)
    pinbuffer1 = PinBuffer(pin=17, T=T, dt=dt)
    pinbuffer2 = PinBuffer(pin=27, T=T, dt=dt)
    output = Integrator(inputs=[pinbuffer1, pinbuffer2], T=T, dt=dt, trigger=0.2, outpin=23, delay=0., weights=[0.5,0.5], func=kernel)

    def update(x):
        for i in x: i.update()

    try:
        counter = 0
        now = time.time()
        while True: 
            counter += 1
            if counter%100 == 0: 
                newtime = time.time()
                print(100/(newtime-now))
                now = newtime
            update([pinbuffer1, pinbuffer2, output])
            time.sleep(dt)
    except KeyboardInterrupt:
        close()