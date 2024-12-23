from PyQt5 import QtWidgets, QtGui,QtCore
from PyQt5.QtWidgets import QScrollArea,QFileDialog,QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QAction,QToolBar,QVBoxLayout
from PyQt5.QtGui import QPalette, QColor,QImage
from PyQt5.QtCore import Qt
from vmbpy import *
import cv2
from components.Capture import GigEVisionCamera, ImageProcess
from components.USBIO import USBIO
from windows.OutputSettingView import OutputSetting
from windows.RecorderView import RecorderView
from windows.SummaryWindow import SummaryWindow

from windows.filmstrip import FilmstripWindow
import cv2
import numpy as np
from threading import Thread

class MainWindow(QtWidgets.QWidget):
    def __init__(self,dbconn,image_process1:ImageProcess,capture1:GigEVisionCamera,recorder1, image_process2:ImageProcess,capture2:GigEVisionCamera,recorder2):
        super(MainWindow, self).__init__()
        self.dbconn=dbconn
        self.image_process1= image_process1
        self.image_process2 = image_process2
        verticalLayout = QtWidgets.QHBoxLayout()
        verticalLayout.setContentsMargins(0,0,0,0)
        verticalLayout.setSpacing(0)
        # palette = self.palette()
        # palette.setColor(QPalette.Window, QColor(0, 0, 0))  # Set the background color to black
        # self.setPalette(palette)
        # self.setAutoFillBackground(True)
        #cam 1
        self.cam_view_1 = CameraView(image_process1,capture1,recorder1,self.showSummary)
        self.cam_view_2 = CameraView(image_process2,capture2,recorder2,self.showSummary)

        verticalLayout.addWidget(self.cam_view_1,1)
        verticalLayout.addWidget(self.cam_view_2,1)
        self.setLayout(verticalLayout)
        USBIO.instance()
        USBIO.StartScan()
        USBIO.instance().frameReady1.connect(capture1.camera_triggered)
        USBIO.instance().frameReady2.connect(capture2.camera_triggered)

    @QtCore.pyqtSlot()
    def showSummary(self):
        self.w1 = SummaryWindow(self.image_process1,self.image_process2)
        self.w1.show()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image : QImage):
        self.image_label.resize(self.scrollArea1.size())
        #self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.image_label.size(),
                                                                          QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation))

class ImageWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.view = QGraphicsView(self.scene)
        #self.setCentralWidget(self.view)

        self.create_toolbar()

        #self.setWindowTitle('Image Viewer')
        #self.setGeometry(300, 300, 800, 600)
    def load_image(self,pixmap):
        self.pixmap_item.setPixmap(pixmap)

    def create_toolbar(self):
        zoom_in_action = QAction('Zoom In', self)
        zoom_out_action = QAction('Zoom Out', self)
        fit_to_window_action = QAction('Fit to Window', self)

        zoom_in_action.triggered.connect(self.zoom_in)
        zoom_out_action.triggered.connect(self.zoom_out)
        fit_to_window_action.triggered.connect(self.fit_to_window)

        toolbar = QToolBar('Image Toolbar')
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(fit_to_window_action)
        
        layout = QVBoxLayout()
        layout.setMenuBar(toolbar)

        layout.addWidget(self.view)

        self.setLayout(layout)

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1 / 1.2, 1 / 1.2)

    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

class CameraView(QtWidgets.QWidget):
    def __init__(self,image_process:ImageProcess,capture:GigEVisionCamera,recorder,openSummary):
        super(CameraView, self).__init__()
        self.image_process = image_process
        self.capture= capture
        self.recorder= recorder
        verticalLayout = QtWidgets.QVBoxLayout()
        self.toolPanel = QtWidgets.QWidget()
        self.topLayout = QtWidgets.QHBoxLayout()

        verticalLayout.setContentsMargins(0,0,0,0)
        verticalLayout.setSpacing(0)

        self.openImgBtn = QtWidgets.QPushButton("Open Image")
        self.openImgBtn.clicked.connect(self.open_image)
        self.topLayout.addWidget(self.openImgBtn)

        self.triggerBtn = QtWidgets.QPushButton("Trigger")
        self.triggerBtn.clicked.connect(self.software_trigger)
        self.topLayout.addWidget(self.triggerBtn)

        self.openFilmstripBtn = QtWidgets.QPushButton("Filmstrip")
        self.openFilmstripBtn.clicked.connect(self.open_filmstrip)
        self.topLayout.addWidget(self.openFilmstripBtn)

        self.openInspectionSetting = QtWidgets.QPushButton("Setting")
        self.openInspectionSetting.clicked.connect(openSummary)
        self.topLayout.addWidget(self.openInspectionSetting)

        self.openRecordSetting = QtWidgets.QPushButton("Record")
        self.openRecordSetting.clicked.connect(self.open_record_setting)
        self.topLayout.addWidget(self.openRecordSetting)

        self.openOutputSetting = QtWidgets.QPushButton("I/O")
        self.openOutputSetting.clicked.connect(self.show_output_setting)
        self.topLayout.addWidget(self.openOutputSetting)

        self.toolPanel.setLayout(self.topLayout)
        #self.toolPanel.setFixedHeight(80)
        #cam 1
        #self.image_label = QtWidgets.QLabel()      
        #self.image_label.setGeometry(0, 0, 780, 562)
        #self.image_label.setFixedSize(900, 700)
        # self.setLayout(verticalLayout)
        # image_frame = QtWidgets.QFrame(self)
        # image_frame.setFrameShape(QtWidgets.QFrame.Shape.Box)
        # image_frame.setLayout(verticalLayout)
        
        self.scrollArea = ImageWindow()
        #self.scrollArea.setWidget(self.image_label)

        verticalLayout.addWidget(self.toolPanel)
        verticalLayout.addWidget(self.scrollArea,1)

        
        self.setLayout(verticalLayout)
        #self.image_label.resize(self.size())

        self.image_process.imageReady.connect(self.setImage)        
    def show_output_setting(self):
        self.w2 = OutputSetting()
        self.w2.show()    
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select image', '', 'Images (*.png *.bmp *.jpg)')
        if file_name:
            image = cv2.imread(file_name)
            self.capture.frameReady.emit(image)
    def software_trigger(self):
        self.capture.trigger()

    def open_setting(self):
        self.w1 = SummaryWindow()
        self.w1.show()
    def open_record_setting(self):
        self.w = RecorderView(self.recorder)
        self.w.show()
    
    def open_filmstrip(self):
        self.w = FilmstripWindow(self.open_image_file)
        self.w.show()

    def open_image_file(self,file_name):
        if file_name:
            image = cv2.imread(file_name)
            if image is not None:
                self.capture.frameReady.emit(image)

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image : QImage):
        #self.image_label.resize(self.scrollArea.size())
        self.scrollArea.load_image(QtGui.QPixmap.fromImage(image))
        # self.image_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.image_label.size(),
        #                                                                   QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation)) 