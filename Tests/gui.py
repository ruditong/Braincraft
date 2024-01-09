'''
gui.py

Implements user interface for interacting with electrical circuit
'''

import sys
import time
import re, random
from utils import *
import numpy as np
import matplotlib.pyplot as pl
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QApplication, QVBoxLayout, 
    QComboBox, QStackedLayout, QFormLayout, QLineEdit,
    QTabWidget,QHBoxLayout, QCheckBox, QPushButton,
    QMainWindow, QStatusBar, QRadioButton, QLabel,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsLineItem)

def find_numbers(s):
    '''Given a string s, return a list of all numbers'''
    n = [float(i) for i in re.findall(r"[-+]?(?:\d*\.*\d+)", s)]
    return n

# Keep track of used pins
INPINS = []

class MainWindow(QMainWindow):
    def __init__(self, app, dt=0.01):
        super().__init__()
        self.app = app
        self.dt = dt
        self.setWindowTitle("BrainCraft Visualizer")

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"FPS = 0.0 Hz")

        # Create gui
        self.gui = GUI(app=app, dt=dt, statusbar=self.statusBar)  
        self.setCentralWidget(self.gui)

    def run(self):
        self.show()
        self.gui.timer.start(int(self.dt*1000))
        self.app.exec_()

class GUI(QWidget):
    def __init__(self, app, statusbar, dt=0.01):
        super().__init__()
        self.dt = dt
        self.app = app
        self.statusBar = statusbar
        self.buffers = []

        # Create a Gridlayout
        layout = QGridLayout()
        # Add widgets
        self.painter = Painter()
        layout.addWidget(self.painter, 0, 0)

        # Add Info bar
        self.info = self.createInfo()
        layout.addWidget(self.info, 0, 1, 2, 1)

        # Add dynamic plotter
        self.plotter = DynamicPlotter(dt=0.01)
        layout.addWidget(self.plotter, 1, 0)

        # Scale the widgets correctly
        layout.setColumnStretch(0, 6)
        layout.setColumnStretch(1, 3)
        layout.setRowStretch(0, 4)
        layout.setRowStretch(1, 3)

        self.setLayout(layout)

        # Set up simulation clock
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateSimulation)

        # Misc attributes
        self.time = 0
        self.counter = 0
        self.skipframes = 10

    def createInfo(self):
        '''Create an InfoBar widget'''
        info = InfoBar()
        self.constructorWindow = None
        info.simulation.apply.clicked.connect(self._apply_button)
        info.simulation.construct.clicked.connect(self._construct_button)
        info.parameters.updatebutton.clicked.connect(self._update_button)
        return info

    def _apply_button(self):
        '''Apply new settings fo the plotter'''
        # Retrieve parameters
        T = float(self.info.simulation.T.text())
        dt = float(self.info.simulation.dt.text())
        # Check that values are proper
        try:
            assert (T > 0) & (dt > 0.001) & (T > dt) 
            assert (T/dt < 1e5)
        except AssertionError:
            return
        
        # Update plotter
        self.plotter.setParameters(T=T, dt=dt)

        # Update timer
        self.dt = dt
        self.timer.stop()
        self.timer.start(int(self.dt*1000))

    def _construct_button(self):
        '''Open new window to create a new neuron'''
        if self.constructorWindow is None:
            self.constructorWindow = NeuronConstructor()
            self.constructorWindow.submit.clicked.connect(self._construct_submit)
            self.constructorWindow.closebutton.clicked.connect(self._construct_close)
        self.constructorWindow.show()

    def _visible_check(self, val):
        '''Toggle buffer on and off'''
        # Get current page ID
        id = self.info.parameters.pageCombo.currentIndex()
        # Toggle visibility in Curves and update plotter y axis
        self.plotter.curves[id].toggleVisibility(val)
        self.plotter.setYrange()

    def _construct_submit(self):
        '''Process constructor and close the window'''
        # Retrieve information
        
        try: params = self.constructorWindow.getParams()
        except AssertionError: 
            print("Could not generate neuron - Check parameters!")
            return

        if self.constructorWindow.radio_input.isChecked(): 
            typ = 'Input'
        elif self.constructorWindow.radio_integrator.isChecked(): 
            typ = 'Integrator'
        
        # Now create a databuffer for the neuron
        T = float(self.info.simulation.T.text())
        dt = float(self.info.simulation.dt.text())
        if params['type'] == 'Input':
            buffer = PinBuffer(pin=int(params['inpin']), T=T, dt=dt)
            buffer.visible = params['visible']

        elif params['type'] == 'Integrator':
            inputs = find_numbers(params['neuron'])
            inputbuffers = [self.buffers[int(i)] for i in inputs]
            weights = find_numbers(params['weights'])
            delays = find_numbers(params['delays'])
            buffer = Integrator(inputs=inputbuffers, T=T, dt=dt, delay=delays, weights=weights)
            buffer.visible=params['visible']

        self.buffers.append(buffer)
        self.plotter.addDataStream(buffer, color=np.array(pl.cm.tab10(len(self.buffers)-1))[:-1]*255)

        # Add a page to parameters
        page = self.info.parameters.addPage(name=params['name'], typ=params['type'], params=params, readonly=True, id=len(self.buffers)-1)
        
        # Connect page buttons
        page.outputs['visible'].toggled.connect(lambda: self._visible_check(page.outputs['visible'].isChecked()))

        # Now call update button to apply all parameters
        self.updateBuffer(len(self.buffers)-1)

    def _construct_close(self):
        '''Close he constructor'''
        # Close the window
        self.constructorWindow.close()
        self.constructorWindow = None

    def _update_button(self):
        '''Update a neuron'''
        if len(self.buffers) == 0: return 
        id = self.info.parameters.pageCombo.currentIndex()
        self.updateBuffer(id)

    def updateBuffer(self, id):
        '''Update the buffer specified in id'''
        params = self.info.parameters.stackedLayout.widget(id).getParams()
        
        # Now go through params and change the buffer accordingly
        # Check for trigger
        trigger, outpin, threshold = params['trigger'], params['outpin'], params['threshold']
        if trigger: self.buffers[id].setTrigger(float(threshold), int(outpin))
        else: self.buffers[id].setTrigger(None, None)

        # Check for functions
        relu, sigmoid, invert, kernel = params['relu'], params['sigmoid'], params['invert'], params['kernel']
        func = lambda x, *y: x
        if kernel:
            tau = float(params['tau'])/self.dt
            func = expKernel(tau=tau)
        self.buffers[id].setFunc(func=func)
        self.buffers[id].relu = relu
        self.buffers[id].invert = invert

        # Check for integrator parameters
        if params['type'] == 'Integrator':
            # Inputs
            inputs = find_numbers(params['neuron'])
            inputbuffers = [self.buffers[int(i)] for i in inputs]
            self.buffers[id].setInputs(inputbuffers)
            # Weights
            weights = find_numbers(params['weights'])
            self.buffers[id].setWeights(weights)
            # Delays
            delays = find_numbers(params['delays'])
            self.buffers[id].setDelays(delays)

    def updateSimulation(self):
        '''Update plots'''
        # Only update plotter if "Run"-checkbox is checked
        if self.info.simulation.checkBox.isChecked():
            self.plotter.updateSimulation()

            # Display update FPS
            self.counter += 1
            if self.counter%self.skipframes == 0:
                time_now = time.time()
                freq = self.skipframes/(time_now - self.time)
                self.time = time_now
                self.counter = 0
                self.statusBar.showMessage(f"FPS = {freq: .1f} Hz")

        self.app.processEvents()

class Painter(QGraphicsView):
    '''Visualizer for neuronal connectivity. Mirrors the simulation setup and allow for user interaction'''
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.circles = []
        self.lines = []

        self.addCircle("Start", pos=(-10,-10))
        self.addCircle("End", pos=(50,50))
        # self.addLine(self.circles[0], self.circles[1])
        

    def addCircle(self, label='', pos=(0,0), radius=50):
        '''Add a circle with label and position'''
        circle = Circle(label, QtCore.QRectF(pos[0], pos[1],radius, radius))
        self.circles.append(circle)
        self.scene.addItem(self.circles[-1])

    def addLine(self, start, end):
        '''Add a connection between cicrles'''
        line = Line(start, end)
        self.lines.append(line)
        self.scene.addItem(self.lines[-1])

    def paintEvent(self, event):
        '''Define pain event handler'''
        super().paintEvent(event)
        
class Circle(QGraphicsEllipseItem):
    '''Circle representing simulation elements'''
    def __init__(self, label='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlags(
            self.flags()
            | QGraphicsItem.ItemSendsGeometryChanges
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
        )
        self.label = label  
        self.setPen(QtGui.QPen(QtGui.QColor("black"), 5))
    
    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.ItemPositionChange
        ):
            return value
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=0):
        super().paint(painter, option, widget)
        painter.drawText(self.boundingRect(), QtCore.Qt.AlignCenter, self.label)

class Line(QGraphicsLineItem):
    '''Line representing connection between neurons'''
    def __init__(self, start, end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlags(
            self.flags()
            | QGraphicsItem.ItemSendsGeometryChanges
            | QGraphicsItem.ItemIsSelectable
        )
        self.setPen(QtGui.QPen(QtGui.QColor("black"), 5))
        self.addItem(start)
        self.adItem(end)
        self.setHandlesChildEvents(True)
        self.start = start
        self.end = end

    def paint(self, painter, option, widget=0):
        self.setLine(self.start.rect().x(), self.start.rect().y(), self.end.rect().x(), self.end.rect().y())
        super().paint(painter, option, widget)

class Parameters(QWidget):
    '''Window to display and change parameters'''
    def __init__(self):
        super().__init__()
        # Container for pages
        self.pages = {}

        # Create top level vertical layout
        layout = QVBoxLayout()

        # Create the stacked layout that contains the parameter pages
        self.stackedLayout = QStackedLayout()

        # Create combo box, i.e. pages
        self.pageCombo = QComboBox()

        # Connect page switch event
        self.pageCombo.activated.connect(self.switchPage)

        # Add layouts to top level layout
        layout.addWidget(self.pageCombo)
        layout.addLayout(self.stackedLayout)

        # Create update button
        self.updatebutton = QPushButton("Update settings")
        layout.addWidget(self.updatebutton)

        self.setLayout(layout)

    def addPage(self, name, typ, params={}, readonly=False, id=None):
        '''Add a page to combo box'''
        # Create new page and add to stack layout
        self.pages[name] = ParameterPage(type=typ, params=params, id=id)
        self.pages[name].outputs['name'].setReadOnly(readonly)
        if typ == 'Input': self.pages[name].outputs['inpin'].setReadOnly(readonly)
        self.stackedLayout.addWidget(self.pages[name])

        # Add page to dropdown
        self.pageCombo.addItem(name)
        return self.pages[name]

    def switchPage(self):
        '''Switch between different pages'''
        self.stackedLayout.setCurrentIndex(self.pageCombo.currentIndex())

class ParameterPage(QWidget):
    '''Parameter information pages'''
    def __init__(self, type, params={}, id=None):
        super().__init__()
        self.type = type
        self.outputs = {}
        self.id = id

        # Depending on the type, create different page formats
        layout = QVBoxLayout()
        if id is None: layout.addWidget(QLabel(f'Type: {self.type}'), 1)
        else: layout.addWidget(QLabel(f"Type: {self.type}\tID: {id}"), 1)

        pagelayout = QFormLayout()

        # Textbox parameters
        self.outputs['name'] = QLineEdit(params.get('name', ''))
        pagelayout.addRow(f"Name", self.outputs['name'])
        self.outputs['outpin'] = QLineEdit(params.get('outpin', ''))
        pagelayout.addRow(f"Output pin", self.outputs['outpin'])
        self.outputs['threshold'] = QLineEdit(params.get('threshold', '0.5'))
        pagelayout.addRow(f"Trigger threshold", self.outputs['threshold'])
        self.outputs['tau'] = QLineEdit(params.get('tau', '0.05'))
        pagelayout.addRow(f"Kernel tau", self.outputs['tau'])

        if self.type == 'Input':
            self.outputs['inpin'] = QLineEdit(params.get('inpin', ''))
            pagelayout.addRow(f"Input pin", self.outputs['inpin'])
            
        elif self.type == 'Integrator':
            self.outputs['neuron'] = QLineEdit(params.get('neuron', ''))
            pagelayout.addRow(f"Input neurons", self.outputs['neuron'])

            self.outputs['weights'] = QLineEdit(params.get('weights', '1'))
            pagelayout.addRow(f"Weights", self.outputs['weights'])

            self.outputs['delays'] = QLineEdit(params.get('delays', '0'))
            pagelayout.addRow(f"Delays", self.outputs['delays'])
            
        pagelayout.setVerticalSpacing(1)
        layout.addLayout(pagelayout, 5) 

        # Checkbox parameters
        checklayout = QGridLayout()
        self.outputs['invert'] = QCheckBox("Invert")
        if params.get('invert', False): self.outputs['invert'].setChecked(True)
        checklayout.addWidget(self.outputs['invert'], 0, 0)

        self.outputs['relu'] = QCheckBox("ReLU")
        if params.get('relu', False): self.outputs['relu'].setChecked(True)
        checklayout.addWidget(self.outputs['relu'], 1, 0)

        self.outputs['sigmoid'] = QCheckBox("Sigmoid")
        if params.get('sigmoid', False): self.outputs['sigmoid'].setChecked(True)
        checklayout.addWidget(self.outputs['sigmoid'], 0, 1)

        self.outputs['kernel'] = QCheckBox("Kernel")
        if params.get('kernel', False): self.outputs['kernel'].setChecked(True)
        checklayout.addWidget(self.outputs['kernel'], 1, 1)

        self.outputs['trigger'] = QCheckBox("Trigger")
        if params.get('trigger', False): self.outputs['trigger'].setChecked(True)
        checklayout.addWidget(self.outputs['trigger'], 2, 0)

        self.outputs['visible'] = QCheckBox("Show")
        if params.get('visible', True): self.outputs['visible'].setChecked(True)
        checklayout.addWidget(self.outputs['visible'], 2, 1)
              
        checklayout.setSpacing(0)
        layout.addLayout(checklayout, 5)
        self.setLayout(layout)
    
    def getParams(self):
        '''Output the data in all widgets'''
        global INPINS
        output = {'type': self.type}
        for key in self.outputs.keys():
            if type(self.outputs[key]) is QLineEdit: output[key] = self.outputs[key].text()
            elif type(self.outputs[key]) is QCheckBox: output[key] = self.outputs[key].isChecked()
        
        # Check parameters for validity
        if output.get('inpin', None) is not None: 
            assert output.get('inpin').isdigit()
            assert (int(output.get('inpin')) >=0) & (int(output.get('inpin')) <= 27)

        if output.get('trigger', False):
            assert output.get('outpin').isdigit()
            assert (int(output.get('outpin')) >=0) & (int(output.get('outpin')) <= 27)


        if output.get('outpin').isdigit():
            # For inputs test if outpin is inpin
            assert int(output.get('outpin')) not in INPINS
            if output['type'] == 'Input':
                assert int(output.get('outpin')) != int(output.get('inpin'))
                INPINS.append(int(output.get('inpin')))

        # For integrator, check for recursions
        if output['type'] == 'Integrator':
            # Retrieve inputs
            inputs = [int(i) for i in find_numbers(output['neuron'])]
            assert self.id not in inputs

        return output

class Simulation(QWidget):
    '''Container for simulation parameters'''
    def __init__(self):
        super().__init__()
        # Create top level layout
        layout = QVBoxLayout()

        # Create Checkbox and reset button
        controllayout = QHBoxLayout()
        self.checkBox = QCheckBox("Run")
        self.reset = QPushButton("Reset")
        controllayout.addWidget(self.checkBox)
        controllayout.addWidget(self.reset)

        # Create form layout
        self.pagelayout = QFormLayout()
        self.T = QLineEdit("1")
        self.dt = QLineEdit("0.01")
        self.pagelayout.addRow(f"Time (s):", self.T)
        self.pagelayout.addRow(f"dt (s):", self.dt)

        # Create Constructor button
        self.construct = QPushButton("Create new neuron")

        # Create apply button
        self.apply = QPushButton("Apply settings")

        layout.addLayout(controllayout)
        layout.addSpacing(20)
        layout.addLayout(self.pagelayout)
        layout.addSpacing(10)
        layout.addWidget(self.construct)
        layout.addSpacing(10)
        layout.addWidget(self.apply)
        self.setLayout(layout)

class InfoBar(QWidget):
    '''Information bar containing parameter tab and simulation tabs'''
    def __init__(self):
        super().__init__()
        # Create top level vertical layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget with two pages for simulation and parameters
        tabs = QTabWidget() 
        self.simulation = Simulation()
        self.parameters = Parameters()
        tabs.addTab(self.simulation, "Simulation")
        tabs.addTab(self.parameters, "Parameters")

        layout.addWidget(tabs)

class DynamicPlotter(pg.GraphicsLayoutWidget):
    '''Realtime visualization of GPIO inputs.'''
    def __init__(self, dt=0.01, skipframes=10):
        super().__init__()
        # Some parameters
        self.skipframes = skipframes
        self.dt = dt

        # Set up plot container
        self.plot = self.addPlot()
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'Neuron ID', '')
        self.plot.setLabel('bottom', 'time', 's')
        self.plot.setYRange(-1, 1, padding=0)
        self.plot.hideButtons()
        self.show()

        # Prepare container for curves
        self.curves = []

    def updateSimulation(self):
        '''Retrieve real time data and update plot'''
        offset = 0
        for i, curve in enumerate(self.curves): 
            curve.update(offset=2*offset)
            if curve.buffer.visible: offset += 1

    def addDataStream(self, buffer, color=np.random.randint(0, 255, size=(3,))):
        '''Add a datastream to plot'''
        newcurve = Curve(canvas=self.plot, buffer=buffer, color=color)
        newcurve.curve.setVisible(buffer.visible)
        self.curves.append(newcurve)
        self.setYrange()

    def setParameters(self, T, dt):
        '''Apply new parameters to plotter'''
        self.dt = dt
        # Loop through each curve and update the x array and the buffers
        for curve in self.curves:
            curve.buffer.setParameters(T, dt)
            curve.x = np.linspace(-curve.buffer.T, 0., curve.buffer._bufsize)

    def setYrange(self):
        '''Update yrange to fit all plots'''
        ncurves = np.array([curve.buffer.visible for curve in self.curves]).sum()
        self.plot.setYRange(-1, 2*(ncurves-1)+1.5, padding=0)

class Curve():
    '''Curve for ploatting datastreams'''
    def __init__(self, canvas, buffer, color=np.random.randint(0, 255, size=(3,))):
        # Set up buffers for data arrays
        self.buffer = buffer
        self.x = np.linspace(-self.buffer.T, 0., self.buffer._bufsize)
        self.canvas = canvas

        self.curve = self.canvas.plot(self.x, self.buffer.databuffer, pen=color)

    def update(self, offset):
        '''Update the plot for one time step'''
        self.buffer.update()
        if self.buffer.visible: self.curve.setData(self.x, np.array(self.buffer.databuffer)+offset)

    def toggleVisibility(self, val):
        '''Toggle visiblity of buffer on and off'''
        self.buffer.visible = val
        self.curve.setVisible(val)

class NeuronConstructor(QWidget):
    '''Constructer window for creating new neurons'''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create a neuron..")
        layout = QGridLayout()

        # Parameter window
        self.parameters = QStackedLayout()
        self.inputParams = ParameterPage(type='Input')
        self.integratorParams = ParameterPage(type='Integrator')
        self.parameters.addWidget(self.inputParams)
        self.parameters.addWidget(self.integratorParams)
        layout.addLayout(self.parameters, 1, 0, 1, 2)

        # Radiobutton to choose the type of neuron to create
        neurontype = QHBoxLayout()
        self.radio_input = QRadioButton("Input")
        self.radio_integrator = QRadioButton("Integrator")
        neurontype.addWidget(self.radio_input)
        neurontype.addWidget(self.radio_integrator)
        self.radio_input.toggled.connect(lambda: self.update_selected_option(0))
        self.radio_integrator.toggled.connect(lambda: self.update_selected_option(1))
        self.radio_input.setChecked(True)
        layout.addLayout(neurontype, 0, 0, 1, 2)

        # Add submit button
        self.submit = QPushButton("Create neuron")
        layout.addWidget(self.submit, 2, 0)

        # Add close button
        self.closebutton = QPushButton("Close")
        layout.addWidget(self.closebutton, 2, 1)

        self.setLayout(layout)

    def update_selected_option(self, option):
        self.parameters.setCurrentIndex(option)

    def getParams(self):
        '''Call the getParams function of the current page'''
        return self.parameters.currentWidget().getParams()


if __name__ == '__main__':
    setup()
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget{font-size: 11pt;}")
    window = MainWindow(app)
    window.run()
    close()
    sys.exit()