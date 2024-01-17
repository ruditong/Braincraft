'''
gui.py

Implements user interface for interacting with electrical circuit
'''

import sys, os, pathlib
import time, math
import re, random
import numpy as np
import matplotlib.pyplot as pl
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QApplication, QVBoxLayout, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsEllipseItem)

import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class Player(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()
        self.setPixmap(QtGui.QPixmap('/home/pi/Projects/Braincraft/Games/examplePixmap.png'))
        self.setOffset(-16,-16)
        self.setTransformationMode(QtCore.Qt.SmoothTransformation)

        self.v = QPointF(0,0)
        self.a = QPointF(0,0)
        self.m = 0.001
        self.friction = 0.2
        self.dt = 1/60.

    def updatePos(self, dt):
        '''Update Player position'''
        self.v = self.v + self.a*self.dt 
        newPos = self.pos() + self.dt*self.v
        if (newPos.x() < -250) or (newPos.x() > 250): self.v.setX(self.v.x()*-1)
        if (newPos.y() < -250) or (newPos.y() > 250): self.v.setY(self.v.y()*-1)
        self.setPos(newPos)

    def updateA(self, F):
        '''Update player velocity'''
        self.a = F/self.m + (-self.v * self.friction)

class PacMan(QGraphicsEllipseItem):
    def __init__(self):
        super().__init__()
        self.setRect(QRectF(-10,-10,20,20))
        self.setSpanAngle(270*16)
        self.setStartAngle(45*16)

        self.pen = QtGui.QPen(QtGui.QColor(242,216,48), 1)
        self.brush = QtGui.QBrush(QtGui.QColor(242,216,48))
        self.setPen(self.pen)
        self.setBrush(self.brush)
        self.setRotation(0)
        self.stepCounter = 0

    def paint(self, painter, option, widget=None):
        '''Controls the appearance of the neuron'''
        self.stepCounter += 1
        mouthCycle = (math.sin(self.stepCounter/60)+1)/2
        self.setStartAngle(int(mouthCycle*45*16))
        self.setSpanAngle(int((1-mouthCycle)*90 + 270)*16)
        super().paint(painter, option, widget)

class Scene(QGraphicsScene):
    '''Visualizer for neuronal connectivity. Mirrors the simulation setup and allow for user interaction'''
    def __init__(self):
        super().__init__()
        # self.player = Player()
        # self.addItem(self.player)
        # self.player.setPos(1,0)
        self.W = self.A = self.S = self.D = False

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run)

        self.player2 = PacMan()
        self.addItem(self.player2)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_W: 
            self.W = True
            self.player2.setRotation(-90)
        elif event.key() == QtCore.Qt.Key_S: 
            self.S = True
            self.player2.setRotation(90)
        elif event.key() == QtCore.Qt.Key_A: 
            self.A = True
            self.player2.setRotation(180)
        elif event.key() == QtCore.Qt.Key_D: 
            self.D = True
            self.player2.setRotation(0)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_W: self.W = False
        elif event.key() == QtCore.Qt.Key_A: self.A = False
        elif event.key() == QtCore.Qt.Key_S: self.S = False
        elif event.key() == QtCore.Qt.Key_D: self.D = False

    def run(self):
        pos = self.player2.pos()
        if self.D: self.player2.setPos(pos + QPointF(2, 0))
        elif self.A: self.player2.setPos(pos + QPointF(-2, 0))
        elif self.W: self.player2.setPos(pos + QPointF(0, -2))
        elif self.S: self.player2.setPos(pos + QPointF(0, 2))

class Canvas(QGraphicsView):
    '''Canvas for scene'''
    def __init__(self):
        super().__init__()
        self.sceneRect = QtCore.QRectF(-256,-256,512,512)
        # Scene parrameters
        self.setFrameStyle(0)
        self.scene = Scene()
        self.setScene(self.scene)
        self.setSceneRect(self.sceneRect)
        self.setRenderHint(QtGui.QPainter.Antialiasing)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.canvas.scene.timer.start(int(int(1/60.*1000)))
    sys.exit(app.exec_())