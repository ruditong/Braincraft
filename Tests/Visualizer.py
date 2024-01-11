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

class Neuron(QGraphicsItem):
    '''Draws a neuron that can be moved around'''
    def __init__(self, input=False, rad=20, pos=(0,0), color=(255,0,0), label='', buffer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines = []
        self.input = input
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemSendsScenePositionChanges)
        self.label = label
        self.buffer = buffer

        # Painter parameters
        self.centre = pos
        self.rad = rad
        self.ellipse = QtCore.QRectF(self.centre[0]-self.rad, self.centre[1]-self.rad, 2*self.rad, 2*self.rad)
        self.inactiveColor = np.array([255,255,255])
        self.baseColor = np.array(color)
        self.color = np.array(color)
        self.pen = QtGui.QPen(QtGui.QColor(*self.color), 2)
        self.fontpen = QtGui.QPen(QtCore.Qt.black, 0)
        self.brush = QtGui.QBrush(QtGui.QColor(*self.color))

    def setColor(self, color):
        '''Set the colour of the brush'''
        self.color = color
        self.brush = QtGui.QBrush(QtGui.QColor(self.color[0], self.color[1], self.color[2]))

    def paint(self, painter, option, widget=None):
        '''Controls the appearance of the neuron'''
        # super().paint(painter, option, widget)
        # painter.save()
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawEllipse(self.ellipse)
        painter.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        painter.setPen(self.fontpen)
        painter.drawText(self.ellipse, QtCore.Qt.AlignCenter, self.label)
        # painter.restore()

    def boundingRect(self):
        '''Calculates the bounding box for collision detection'''
        adjust = self.pen.width()/2
        return self.ellipse.adjusted(-adjust, -adjust, adjust, adjust)

    def shape(self):
        '''Define the precise shape of the item'''
        path = QtGui.QPainterPath()
        path.addEllipse(self.ellipse)
        return path
    
    def addLine(self, lineItem):
        for existing in self.lines:
            if existing.endPoints() == lineItem.endPoints():
                # another line with the same control points already exists
                return False
        self.lines.append(lineItem)
        return True

    def removeLine(self, lineItem, scene=True):
        for existing in self.lines:
            if existing.endPoints() == lineItem.endPoints():
                if scene: self.scene().removeItem(existing)
                self.lines.remove(existing)
                return True
        return False
    
    def itemChange(self, change, value):
        '''Move all lines attached to this item'''
        for line in self.lines:
            line.updateLine(self)
        return super().itemChange(change, value)
    
    def updateAppearance(self):
        '''Update color'''
        if self.buffer is not None: 
            color = lerp(self.inactiveColor, self.baseColor, self.buffer.databuffer[-1])
            self.setColor(color)
            self.update()
            return True
        return False

def lerp(x, y, t):
    '''Linearly interpolate between x and y'''
    v = y-x
    return x + t*v

class Connection(QGraphicsLineItem):
    '''Connection between neurons'''
    def __init__(self, start, p2):
        super().__init__()
        self.start, self.end = start, None
        
        # Move line behind neurons
        self.setZValue(-1)

        self.rad = 5.
        self._line = QtCore.QLineF(self.start.scenePos(), p2)
        self.custompen = QtGui.QPen(QtGui.QColor("black"), 5)
        self.brush = QtGui.QBrush(QtGui.QColor("black"))

    def setP2(self, p2):
        self._line.setP2(p2)
        self.setLine(self._line)

    def setStart(self, start):
        self.start = start

    def setEnd(self, end):
        self.end = end
        self.updateLine(end)
        self.update()

    def updateLine(self, source):
        if source == self.start:
            self._line.setP1(source.scenePos())
        else:
            self._line.setP2(source.scenePos())
        self.setLine(self._line)

    def endPoints(self):
        '''Return start and end points'''
        return self.start, self.end

    def paint(self, painter, option, widget=None):
        '''Overwrite paint to add arrowhead'''
        painter.setPen(self.custompen)
        painter.setBrush(self.brush)
        painter.drawLine(self._line)

        # Draw arrow head - if end exist, shift the arrow head back
        if self.end is None:
            painter.drawEllipse(QtCore.QRectF(self._line.p2()-QtCore.QPointF(self.rad, self.rad), 
                                              QtCore.QSizeF(self.rad*2,self.rad*2)))
        else:
            endpoint = self.end.pos() - QtCore.QPointF(self._line.unitVector().dx()*(20.+self.rad),
                                                      self._line.unitVector().dy()*(20.+self.rad)) -QtCore.QPointF(self.rad, self.rad)
            painter.drawEllipse(QtCore.QRectF(endpoint, 
                                              QtCore.QSizeF(self.rad*2,self.rad*2)))

    def shape(self):
        '''Calculates the bounding box for collision detection'''
        path = QtGui.QPainterPath()
        polygon = QtGui.QPolygonF()
        # Add four corner points
        adjust = self.rad*2
        normal = QtCore.QPointF(self._line.normalVector().dx(), self._line.normalVector().dy())
        polygon << (self._line.p1()+normal*adjust) << (self._line.p1()-normal*adjust) << (
                    self._line.p2()+normal*adjust) << (self._line.p2()-normal*adjust)
        path.addPolygon(polygon)
        return path

class Scene(QGraphicsScene):
    '''Visualizer for neuronal connectivity. Mirrors the simulation setup and allow for user interaction'''
    def __init__(self):
        super().__init__()
        # Neuron parameters
        self.rad = 20
        self.neurons = []

        # Event handling variables
        self.start = None
        self.connect = None

        # Tests
        # self.addNeuron(pos=(100,100), label='1')
        # self.addNeuron(pos=(-100,-100), label='2')

    def addNeuron(self, pos, color=(255,0,0), label='', buffer=None):
        '''Add neuron to canvas and move to position pos'''
        neuron = Neuron(rad=self.rad, color=color, label=label, buffer=buffer)
        neuron.setPos(*pos)
        self.addItem(neuron)
        self.neurons.append(neuron)

    def updateNeurons(self):
        '''Update the appearance of neurons'''
        for neuron in self.neurons:
            neuron.updateAppearance()

    def mousePressEvent(self, event):
        '''On mouse press, if mouse is on neuron, create connection from that neuron'''
        # if event.button() == QtCore.Qt.LeftButton:
        #     pos = event.scenePos()
        #     item = self.itemAt(pos, QtGui.QTransform())
        #     print(item, item.pos(), item.scenePos())
        if event.button() == QtCore.Qt.RightButton:
            pos = event.scenePos()
            item = self.itemAt(pos, QtGui.QTransform())
            if isinstance(item, Neuron):
                self.start = item
                self.connect = Connection(start=item, p2=pos)
                self.addItem(self.connect)
                self.start.setFlag(QGraphicsItem.ItemIsMovable, False)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        '''If mouse was pressed and connections created, update the connection end point'''
        if self.connect:
            # Implement some snapping
            item = self.itemAt(event.scenePos(), QtGui.QTransform())
            if isinstance(item, Neuron) and (item is not self.start):
                pos = item.scenePos()
            else:
                pos = event.scenePos()
            self.connect.setP2(pos)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        '''Create or destroy connection'''
        if self.connect:
            pos = event.scenePos()
            item = self.itemAt(pos, QtGui.QTransform())
            if isinstance(item, Neuron) and (item != self.start):
                self.connect.setEnd(item)
                if self.start.addLine(self.connect):
                    item.addLine(self.connect)
                    
                else:
                    # If line was not created successfully, it means that line already exists
                    # In this case, delete the preexisting line
                    self.connect.start.removeLine(self.connect)
                    self.connect.end.removeLine(self.connect, scene=False)
                    self.removeItem(self.connect)
                             
            else:
                # self.start.removeLine(self.connect)
                self.removeItem(self.connect)
                
            self.start.setFlag(QGraphicsItem.ItemIsMovable, True)

        self.start = None
        self.connect = None
        super().mouseReleaseEvent(event)

class Canvas(QGraphicsView):
    '''Canvas for scene'''
    def __init__(self, width, height):
        super().__init__()
        # Scene parrameters
        self.setFrameStyle(0)
        self.scene = Scene()
        self.setScene(self.scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        size = event.size()
        self.setSceneRect(-size.width()//2,-size.height()//2,size.width(),size.height())
        self.scene.setSceneRect(-size.width()//2,-size.height()//2,size.width(),size.height())
        self.fitInView(-size.width()//2,-size.height()//2,size.width(),size.height(),QtCore.Qt.KeepAspectRatio)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    canvas = Canvas(1000,500)
    canvas.show()
    sys.exit(app.exec_())
