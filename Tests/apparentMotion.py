'''
Visualize apparent motion experiment.
Display two dot at distance x and delay dt.
'''

import os
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
# Qt imports
from PyQt5.QtWidgets import (QWidget,
                             QApplication,
                             QGridLayout,
                             QFormLayout,
                             QVBoxLayout,
                             QTabWidget,
                             QCheckBox,
                             QTextEdit,
                             QLineEdit,
                             QComboBox,
                             QFileDialog,
                             QSlider,
                             QPushButton,
                             QLabel,
                             QAction,
                             QMenuBar,
                             QGraphicsView,
                             QGraphicsScene,
                             QGraphicsItem,
                             QGraphicsLineItem,
                             QGroupBox,
                             QTableWidget,
                             QMainWindow,
                             QDockWidget,
                             QFileDialog,
                             QGraphicsEllipseItem,
                             QGraphicsSimpleTextItem)
from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor
from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF,QTimer

class ScanboxViewer(QMainWindow):
    app = None
    def __init__(self,app=None, blinkInterval=200.):
        super(ScanboxViewer,self).__init__()
        self.app = app
        self.blinkInterval = blinkInterval
        self.initUI()
        
    def initUI(self):
        # Menu
        self.setWindowTitle("Apparent motion")
        self.tabs = []
        self.widgets = []
        self.tabs.append(QDockWidget("Frames",self))
        self.widgets.append(ImageViewerWidget(self,))

        self.tabs[-1].setWidget(self.widgets[-1])
        self.tabs[-1].setFloating(False)
        self.addDockWidget(
            Qt.RightDockWidgetArea and Qt.TopDockWidgetArea,
            self.tabs[-1])
        self.controlWidget = ControlWidget(self)
        self.tabs.append(QDockWidget("Frame control",self))
        self.tabs[-1].setWidget(self.controlWidget)
        self.tabs[-1].setFloating(False)
        self.addDockWidget(Qt.BottomDockWidgetArea,self.tabs[-1])
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.timerUpdate)
        self.timer.start(self.blinkInterval/2)
        self.move(0, 0)
        # self.setFixedSize(500,500)
        self.show()

    def timerUpdate(self):
        self.widgets[0].update()

class ControlWidget(QWidget):
    def __init__(self,parent):
        super(ControlWidget,self).__init__()	
        self.parent = parent
        form = QFormLayout()
        self.setLayout(form)
        self.intervalSlider = QSlider(Qt.Horizontal)
        self.intervalSlider.setMinimum(0)
        self.intervalSlider.setMaximum(1000)
        self.intervalSlider.setSingleStep(1)
        self.intervalSlider.setValue(self.parent.blinkInterval) 
        
        self.intervalSliderLabel = QLabel('Interval [{} ms]:'.format(self.intervalSlider.value()))
        self.intervalSlider.valueChanged.connect(self.setTimer)
        form.addRow(self.intervalSliderLabel, self.intervalSlider)

        self.distanceSlider = QSlider(Qt.Horizontal)
        self.distanceSlider.setMinimum(1)
        self.distanceSlider.setMaximum(400)
        self.distanceSlider.setSingleStep(1)
        self.distanceSlider.setValue(self.parent.blinkInterval)        
        
        self.distanceSliderLabel = QLabel('Distance [{}]:'.format(self.distanceSlider.value()))
        self.distanceSlider.valueChanged.connect(self.setDistance)
        form.addRow(self.distanceSliderLabel, self.distanceSlider)

    def setTimer(self,value):
        self.intervalSliderLabel.setText('Interval [{0} ms]:'.format(int(value)))
        self.parent.timer.setInterval(value/2)

    def setDistance(self,value):
        self.distanceSliderLabel.setText('Distance [{0}]:'.format(int(value)))
        radius = self.parent.widgets[0].radius
        self.parent.widgets[0].circle1.setRect(-value/2-radius, 0-radius, radius*2, radius*2)
        self.parent.widgets[0].circle2.setRect(value/2-radius, 0-radius, radius*2, radius*2)

class ImageViewerWidget(pg.GraphicsLayoutWidget):
    def __init__(self,parent):
        super(ImageViewerWidget,self).__init__()
        self.parent = parent
        
        p1 = self.addPlot(title="")
        self.setRange(QRectF(0,0,500,500))
        p1.setRange(xRange=[-200,200], yRange=[-200,200])

        # Draw a circle
        self.white = np.array([255,255,255])
        self.pos1, self.pos2 = [-25,0], [25,0]
        self.radius = 20
        self.circle1 = QGraphicsEllipseItem(self.pos1[0]-self.radius, self.pos1[1]-self.radius, self.radius*2, self.radius*2)
        self.brush1 = pg.mkBrush(self.white)
        self.circle1.setPen(pg.mkPen((0, 0, 0, 0)))
        self.circle1.setBrush(self.brush1)
        p1.addItem(self.circle1)

        self.circle2 = QGraphicsEllipseItem(self.pos2[0]-self.radius, self.pos2[1]-self.radius, self.radius*2, self.radius*2)
        self.brush2 = pg.mkBrush(self.white)
        self.circle2.setPen(pg.mkPen((0, 0, 0, 0)))
        self.circle2.setBrush(self.brush2)
        p1.addItem(self.circle2)
        
        self.show()

    def update(self):
        # color1 = self.brush1.color()
        # self.brush2.setColor(color1)
        # self.brush1.setColor(QColor(*(self.white-np.array([color1.red(), color1.green(), color1.blue()]))))
        # self.circle1.setBrush(self.brush1)
        # self.circle2.setBrush(self.brush2)
        self.circle2.setVisible(self.circle1.isVisible())
        self.circle1.setVisible(not(self.circle1.isVisible()))
        

    
app = QApplication(['hi'])
w = ScanboxViewer(app=app)
app.exec_()
