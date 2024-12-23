import os
import sys
from PyQt5.QtWidgets import QScrollArea,QDesktopWidget,QFrame,QMessageBox,QFileDialog, QGridLayout,QLabel,QButtonGroup, QWidget,QHBoxLayout,QBoxLayout,QSpacerItem,QSizePolicy, QVBoxLayout, QPushButton, QStackedWidget,QMainWindow
from PyQt5 import QtCore,QtGui
import cv2
import threading
from components.Capture import Capture, GigEVisionCamera, ImageProcess
from components.Recorder import Recorder
from components.TouchButton import TouchButton
from components.USBIO import USBIO
from components.UserContext import UserContext
from tools.InstanceSegmentation import InstanceSegmentation
from windows.LogWindow import LogWindow
from windows.LoginWindow import LoginWindow
from windows.MainWindow import MainWindow
from windows.RecorderView import RecorderView
from windows.InstaneSegmentationView import InstanceSegmentationView
# from windows.TextRecognitionSettingView import TextRecognitionSettingView
# from windows.TextRecognitionSettingView import TextRecognitionSettingView
from windows.ToolSettingView import ToolSettingView
from windows.camera_settings import CameraSetting
from vmbpy import *


from windows.filmstrip import FilmstripWindow

class App(QMainWindow):
    def __init__(self,dbname):
        super().__init__()
        self.title = 'ID Reader App'
        self.dbname = dbname
        #self.setStyleSheet("background-color: gray;")
        #self.setGeometry(0,0,1600,980)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.showFullScreen()
        self.capture = GigEVisionCamera(cam_id=0,setting_file = 'data/cam1/cam_setting.xml')
        self.image_process = ImageProcess(self.capture,data_dir="data/cam1")

        self.capture2 = GigEVisionCamera(cam_id=1,setting_file = 'data/cam2/cam_setting.xml')
        self.image_process2 = ImageProcess(self.capture2,data_dir="data/cam2")
        #self.image_process1 = ImageProcess(self.capture1)

        save_image_dir = "C:/images1"
        self.recorder = Recorder(self.dbname,save_image_dir,self.image_process)

        save_image_dir2 = "C:/images2"
        self.recorder2 = Recorder(self.dbname,save_image_dir2,self.image_process2)

        self.captureThread = QtCore.QThread(self)
        self.image_processThread = QtCore.QThread(self)
        self.recorderThread = QtCore.QThread(self)

        self.captureThread2 = QtCore.QThread(self)
        self.image_processThread2 = QtCore.QThread(self)
        self.recorderThread2 = QtCore.QThread(self)

        self.captureThread.start()
        self.image_processThread.start()
        self.recorderThread.start()

        self.captureThread2.start()
        self.image_processThread2.start()
        self.recorderThread2.start()

        self.capture.moveToThread(self.captureThread)
        self.image_process.moveToThread(self.image_processThread)
        self.recorder.moveToThread(self.recorderThread)

        self.capture2.moveToThread(self.captureThread2)
        self.image_process2.moveToThread(self.image_processThread2)
        self.recorder2.moveToThread(self.recorderThread2)
        #self.setGeometry(0,0,1900,1080)
        #self.setFixedSize(1400,800)
        self.initUI()

        self.btn_login.clicked.connect(self.login)
        self.btn_logout.clicked.connect(self.logout)
        self.capture.started.connect(lambda: print("started"))
        self.btn_start.clicked.connect(self.StartCapture)
        self.btn_stop.clicked.connect(self.StopCapture)
        self.btn_trigger.clicked.connect(self.capture.trigger)
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_filmstrip.clicked.connect(self.open_filmstrip_window)
        self.btn_quit.clicked.connect(self.btn_quit_clicked)
        self.group_button.buttonClicked.connect(self.group_button_clicked)

        self.user_context = UserContext(self.dbname,'Monitor','MONITOR')
        self.on_logged_in()
        self.prev_nav_button_id = None
        self.home_button.click()

        # this will hide the title bar
        #self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        # self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        # self.showFullScreen()
        self.captureEvent = threading.Event()
        self.threadCapture = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
    def StartCapture(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)        
        self.capture.start()
        self.capture2.start()
        print('vimba stop')
    def StopCapture(self):
        self.capture.stop()
        self.capture2.stop()
        self.captureEvent.set()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 4))
        qp.drawRect(self.rect())
        
        qp.setOpacity(0.01)
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(self.palette().window())
        qp.drawRect(self.rect())
    def closeEvent(self, event):
        super(App, self).closeEvent(event)
        self.on_exit()
        event.accept()

    def on_exit(self):
        print('Closing App ...')
        self.capture.stop()
        self.capture.release()
        self.capture2.stop()
        self.capture2.release()
        self.captureThread.exit()
        self.captureThread2.exit()
        self.image_processThread.exit()
        self.image_processThread2.exit()
        self.recorder.dispose()
        self.recorder2.dispose()
        self.recorderThread.exit()
        self.recorderThread2.exit()
        USBIO.is_scanning = False
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select image', '', 'Images (*.png *.bmp *.jpg)')
        if file_name:
            image = cv2.imread(file_name)
            self.capture.frameReady.emit(image)
    def open_filmstrip_window(self):
        self.w = FilmstripWindow(self.open_image_file)
        self.w.show()
    def open_image_file(self,file_name):
        if file_name:
            image = cv2.imread(file_name)
            if image is not None:
                self.capture.frameReady.emit(image)
    def btn_quit_clicked(self):
        result = QMessageBox.question(self,"Warning",'Do you want to exit program?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        if result == QMessageBox.StandardButton.Yes:
            self.close()

    def group_button_clicked(self,button):
        print(button.text())
        if self.prev_nav_button_id: 
            old_button = self.group_button.button(self.prev_nav_button_id)
            old_button.setFlat(False)
            old_button.setFont(QtGui.QFont('SansSerif',15,QtGui.QFont.Weight.Medium))
        button.setFlat(True)
        button.setFont(QtGui.QFont('SansSerif',15,QtGui.QFont.Weight.Bold))        
        self.prev_nav_button_id = self.group_button.id(button)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.centralwidget = QFrame()
        # self.centralwidget.setFrameShape(QFrame.Shape.Box)
        self.setCentralWidget(self.centralwidget)
        # Create a vertical layout
        layout = QHBoxLayout()
        self.centralwidget.setLayout(layout)
        # Create a stacked widget (for the views)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setContentsMargins(0,0,0,0)
        # self.stacked_widget.setFixedSize(810,580)
        # Create the views
        self.home_view = MainWindow(self.dbname,self.image_process,self.capture,self.recorder,self.image_process2,self.capture2,self.recorder2)
        self.camera_setting_view = QWidget()
        self.recorder_setting_view = RecorderView(self.recorder)
        self.log_viewer_view = LogWindow(self.dbname)
        self.scroll = QScrollArea()
        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        
        self.tool_setting_view = ToolSettingView()
        self.scroll.setWidget(self.tool_setting_view)
        if type(self.image_process.Detection) is InstanceSegmentation:
            text_detection_v2_setting_view = InstanceSegmentationView(self.image_process.Detection)
            self.tool_setting_view.add_view('Instance Segmentation',text_detection_v2_setting_view)
        #self.tool_setting_view.add_view("Text Recognition",TextRecognitionSettingView(self.image_process.Recognition))
        self.tool_setting_view.load_tools()
        # Add the views to the stacked widget
        self.stacked_widget.addWidget(self.home_view)
        self.stacked_widget.addWidget(self.camera_setting_view)
        self.stacked_widget.addWidget(self.recorder_setting_view)
        self.stacked_widget.addWidget(self.log_viewer_view)
        self.stacked_widget.addWidget(self.scroll)

        # Create the navigation buttons
        self.home_button = TouchButton('Home')
        self.home_button.clicked.connect(self.home_button_clicked)
        
        camera_setting_button = TouchButton('Camera Setting')
        camera_setting_button.clicked.connect(self.camera_setting_button_clicked)
        
        # recorder_setting_button = TouchButton('Recorder Setting')
        # recorder_setting_button.clicked.connect(self.recorder_setting_button_clicked)
        
        log_viewer_button = TouchButton('Log Viewer')
        log_viewer_button.clicked.connect(self.log_viewer_button_clicked)

        tool_setting_button = TouchButton('Tool Setting')
        tool_setting_button.clicked.connect(self.tool_setting_button_clicked)

        self.group_button = QButtonGroup()
        self.group_button.addButton(self.home_button)
        #self.group_button.addButton(camera_setting_button)
        #self.group_button.addButton(recorder_setting_button)
        self.group_button.addButton(log_viewer_button)
        #self.group_button.addButton(tool_setting_button)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.btn_login = TouchButton("Login")
        self.btn_logout = TouchButton("Logout")
        self.lb_user = QLabel()
        self.lb_priv = QLabel()
        self.btn_quit = TouchButton('Quit')
        
        self.btn_start = TouchButton('Start')
        self.btn_start.setStyleSheet(u"QPushButton {\n"
        "border-radius: 8px;\n"
        "padding: 1px 5px;\n"
        " color: #000;\n"
        "border: 4px solid rgb(232, 232, 232);\n"
        "  background-color: rgb(255, 255, 0);\n"
        "}\n"
        "\n"
        "QPushButton:hover {\n"
        "  background-color: rgb(255, 236, 16);\n"
        "}\n"
        "\n"
        "QPushButton:disabled {\n"
        "  background-color: rgba(0, 0, 0,30);\n"
        " color: rgba(0, 0, 0,20);\n"
        "}\n"
        "\n"
        "QPushButton:pressed {\n"
        "  border: 4px solid rgb(211, 211, 0);\n"
        "background-color:rgb(234, 234, 0);\n"
        "}")

        self.btn_stop = TouchButton('Stop')
        self.btn_stop.setStyleSheet(u"QPushButton {\n"
        "border-radius: 8px;\n"
        "padding: 1px 5px;\n"
        " color: #0;\n"
        "border: 4px solid rgb(232, 232, 232);\n"
        "  background-color: rgb(255, 85, 0);\n"
        "}\n"
        "\n"
        "QPushButton:hover {\n"
        "  background-color: rgb(255, 128, 1);\n"
        "}\n"
        "\n"
        "QPushButton:disabled {\n"
        "  background-color: rgba(0, 0, 0,30);\n"
        " color: rgba(0, 0, 0,20);\n"
        "}\n"
        "\n"
        "QPushButton:pressed {\n"
        "  border: 4px solid rgb(255, 85, 0);\n"
        "background-color:rgb(234, 78, 0);\n"
        "}")
        self.btn_trigger = TouchButton('Trigger')
        self.btn_open_image = TouchButton('Open Image')
        self.btn_open_filmstrip = TouchButton('Open Filmstrip')
        
        # Add the buttons and the stacked widget to the layout
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.Box)
        left_panel_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        left_panel.setLayout(left_panel_layout)
        

        control_panel_layout = QGridLayout()
        control_panel_layout.addWidget(self.btn_start,0,0)
        control_panel_layout.addWidget(self.btn_stop,1,0)
        #control_panel_layout.addWidget(self.btn_trigger,2,0)
        #control_panel_layout.addWidget(self.btn_open_image,3,0)
        #control_panel_layout.addWidget(self.btn_open_filmstrip,4,0)
        left_panel_layout.addLayout(control_panel_layout)
        
        left_panel_layout.addWidget(self.home_button)
        #left_panel_layout.addWidget(camera_setting_button)
        #left_panel_layout.addWidget(recorder_setting_button)
        left_panel_layout.addWidget(log_viewer_button)
        #left_panel_layout.addWidget(tool_setting_button)

        left_panel_layout.addItem(verticalSpacer)
        
        user_panel_layout = QGridLayout()
        user_panel_layout.addWidget(self.btn_login,3,0)
        user_panel_layout.addWidget(self.btn_logout,4,0)
        user_panel_layout.addWidget(self.lb_user,1,0)
        user_panel_layout.addWidget(self.lb_priv,2,0)
        user_panel_layout.addWidget(self.btn_quit,5,0)
        left_panel_layout.addLayout(user_panel_layout)

        
        layout.addWidget(self.stacked_widget)
        layout.addWidget(left_panel)
        left_panel.setFixedWidth(200)
        #self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    def tool_setting_button_clicked(self):
        if self.prev_nav_button_id == self.group_button.id(self.sender()):
            return
        self.stacked_widget.setCurrentWidget(self.scroll) 

    def log_viewer_button_clicked(self):
        if self.prev_nav_button_id == self.group_button.id(self.sender()):
            return
        self.log_viewer_view.search_data()
        self.stacked_widget.setCurrentWidget(self.log_viewer_view) 

    def recorder_setting_button_clicked(self):
        if self.prev_nav_button_id == self.group_button.id(self.sender()):
            return
        self.stacked_widget.setCurrentWidget(self.recorder_setting_view) 

    def home_button_clicked(self):
        if self.prev_nav_button_id == self.group_button.id(self.sender()):
            return
        self.stacked_widget.setCurrentWidget(self.home_view) 

    def camera_setting_button_clicked(self):
        if self.prev_nav_button_id == self.group_button.id(self.sender()):
            return
        if not self.capture.is_connected:
            QMessageBox.about(self,'Error','Camera is not connected.\nPlease check connection and restart program.')
            return
        # self.prev_nav_button_id = self.group_button.id(self.sender())
        layout = QVBoxLayout()
        self.camera_setting_view.setLayout(layout)
        layout.addWidget(CameraSetting())
        self.stacked_widget.setCurrentWidget(self.camera_setting_view) 

    def logout(self):
        self.user_context.log_out()
        self.on_logged_in()

    def login(self):
        self.wd = LoginWindow(self.user_context)
        self.wd.logged_in.connect(self.on_logged_in)
        self.wd.show()

    def on_logged_in(self):
            self.lb_user.setText("User: {}".format(self.user_context.username))
            self.lb_priv.setText("Privilege: {}".format(self.user_context.role))
            # if self.user_context.is_admin():
            #     print('admin')
            #     self.btn_start.setEnabled(True)
            #     self.btn_stop.setEnabled(True)
            #     self.btn_trigger.setEnabled(True)
            # else:
            #     self.btn_start.setEnabled(False)
            #     self.btn_stop.setEnabled(False)
            #     self.btn_trigger.setEnabled(False)
    