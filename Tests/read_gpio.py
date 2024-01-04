'''
read_gpio.py

Simple class to read GPIO inputs and visualize them in real time.
'''

import collections, time
import numpy as np
import matplotlib.pyplot as pl
import RPi.GPIO as GPIO 
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

def setup():
    '''Set up environment for GPIO'''
    GPIO.setmode(GPIO.BCM)
    return 0

def close():
    '''Close the environment and clean up'''
    GPIO.cleanup()
    return 0

class Plot():
    '''PyQT plot class for real time visualization.'''
    def __init__(self, canvas, buffers, size=(600,350), row=1, col=1, yrange=(-1,2)):
        # Set up buffers for data arrays
        self.buffers = buffers
        self.x = [np.linspace(-buffer.T, 0., buffer._bufsize) for buffer in self.buffers]
        self.y = [np.zeros(buffer._bufsize, dtype=float) for buffer in self.buffers]

        # Set up plot widget
        self.plot = canvas.pglayout.addPlot(row=row, col=col)
        self.plot.resize(*size)
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'GPIO input', '')
        self.plot.setLabel('bottom', 'time', 's')
        self.plot.setYRange(yrange[0],yrange[1],padding=0)

        # Now generate a curve per buffer
        self.curves = [self.plot.plot(self.x[i], self.y[i], pen=np.array(pl.cm.tab10(i))[:-1]*255) for i in range(len(self.y))]

        # Add self to pglayout
        canvas.plots.append(self)

    def update(self):
        '''Update the plot for one time step'''
        for i, buffer in enumerate(self.buffers):
            buffer.update()
            self.curves[i].setData(self.x[i], buffer.databuffer)

class DynamicBuffer():
    '''Buffer class that stores data in real time for plotting purposes'''
    def __init__(self, T, dt, func=lambda: 0, tau=1):
        '''
        Initialise buffer. The buffer is set to size=T/dt.
        Parameters:
            T (float)   : Total time duration to buffer (s)
            dt (float)  : Sample interval (s)
        '''
        self.T, self.dt = T, dt
        self._interval = int(dt*1000)
        self._bufsize = int(T/dt)
        self.databuffer = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.func = func
        self.mean = 0
        self.tau = tau

    def update(self):
        '''Update the buffer'''
        sample = self.func()
        self.mean = self.mean + (sample-self.mean)/self.tau
        self.databuffer.append(max([sample-self.mean, 0]))

class PinBuffer(DynamicBuffer):
    '''Data buffer for pin inputs'''
    def __init__(self, pin, T, dt, func=lambda x, *y: x, tau=1):
        super().__init__(T, dt, tau=tau)
        self.pin = pin

        # Set up the pin
        GPIO.setup(self.pin, GPIO.IN)

        # Read pin input as function
        self.func = func

    def update(self):
        '''Updaet the buffer'''
        sample = self.func(GPIO.input(self.pin), self.databuffer)
        # self.mean = self.mean + (sample-self.mean)/self.tau
        # self.databuffer.append(min([self.mean, 1]))
        self.databuffer.append(sample)
        

class DynamicPlotter():
    '''Realtime visualization of GPIO inputs. Pins is a list of all pins to be shown'''
    def __init__(self, dt=0.01, skipframes=10):
        # Some parameters
        self.skipframes = skipframes

        # PyQT Graph
        self.app = QtWidgets.QApplication([])
        self.pglayout = pg.GraphicsLayoutWidget()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplot)
        self.timer.start(int(dt*1000))

        # Set up buffers for data arrays
        self.plots = []
        self.pglayout.show()

        # Prepare triggers
        self.triggers = []

    def updateplot(self):
        '''Retrieve real time data and update plot'''
        # Loop over plots and update
        self.counter += 1 
        for i, plot in enumerate(self.plots):
            plot.update()
        
        # Update the frequency
        if self.counter%self.skipframes == 0:
            time_now = time.time()
            freq = self.skipframes/(time_now - self.time)
            self.time = time_now
            self.plots[0].plot.setTitle(f"{freq: .1f} Hz")

        self.app.processEvents()

        # Update triggers
        for trigger in self.triggers: trigger.update()

    def run(self):
        self.counter = 0
        self.time = time.time()
        self.app.exec_()

class Trigger():
    '''Trigger an event and send out a TTL'''
    def __init__(self, pin, func):
        '''
        Initialize the class.
        Parameters:
            pin : Pin to send TTL to
            func: Function to detect trigger
        '''
        self.pin = pin
        self.func = func

        # Setup pin
        GPIO.setup(pin, GPIO.OUT)

    def update(self):
        '''Check for trigger'''
        GPIO.output(self.pin, self.func())

def poisson(lam, dt):
    '''Poisson point process with mean l'''
    prob = lam * dt
    val = np.random.rand() < prob
    return val

class expKernel():
    '''Apply an exponential kernel to simulated data mimicking neuron EPSPs'''
    def __init__(self, tau):
        self.tau = tau
        self.x = 0

    def __call__(self, x, *y):
        self.x = self.x + (x-self.x)/self.tau
        return min([self.x, 1])


if __name__ == '__main__':
    setup()

    invert = lambda x: 1-x
    func = expKernel(tau=100)

    T = 1
    dt = 0.01
    pinbuffer1 = PinBuffer(pin=17, T=T, dt=dt, tau=1e9, func=func)
    pinbuffer2 = PinBuffer(pin=27, T=T, dt=dt, tau=1e9)
    comparator1 = DynamicBuffer(T=T, dt=dt, func=lambda: pinbuffer1.databuffer[-int(0.05/dt)] + pinbuffer2.databuffer[-1], tau=1e9)
    comparator2 = DynamicBuffer(T=T, dt=dt, func=lambda: pinbuffer2.databuffer[-int(0.05/dt)] + pinbuffer1.databuffer[-1], tau=1e9)

    reichardt = DynamicBuffer(T=T, dt=dt, func=lambda: ((comparator1.databuffer[-1] > 1.2)*(comparator1.databuffer[-1] - comparator2.databuffer[-1]))*2 , tau=1e9)

    trigger = Trigger(pin=23, func=lambda: (reichardt.databuffer[-1]) > 1)

    plotter = DynamicPlotter(dt=dt, skipframes=10,)
    plotter.triggers.append(trigger)

    plot1 = Plot(plotter, buffers=[pinbuffer1, pinbuffer2], size=(600,200))
    plot2 = Plot(plotter, buffers=[comparator1, comparator2], row=2, yrange=(-1,3), size=(600,200))
    #plot3 = Plot(plotter, buffers=[reichardt], row=3, yrange=(-1,4), size=(600,200))

    #plotter.plots = plotter.plots + [comparator1, comparator2]
    plotter.run()

    close()
        