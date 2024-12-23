import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QWidget, QGridLayout, QGroupBox
from components.custom_spinbox import TouchIntSpinBox

trigger_source_list = ['Freerun','Line1','FixedRate','Software']
trigger_activation_list = ['RisingEdge','FallingEdge','AnyEdge','LevelHigh','LevelLow']
trigger_selector_list = ['FrameStart','AcquisitionStart','AcquisitionEnd','AcquisitionRecord']

trigger_mode_list=['On','Off']
user_default_selector_list = ['Default','UserSet1','UserSet2','UserSet3']
user_selector_list = ['Default','UserSet1','UserSet2','UserSet3']
features = ["ExposureTimeAbs","Gain","TriggerSource","TriggerActivation",
            "TriggerSelector","TriggerMode","UserSetDefaultSelector","UserSetSelector"]
class CameraSetting(QWidget):
    def __init__(self, parent=None):
        super(CameraSetting, self).__init__(parent)
        feature_value = get_feature_value(features)
        self.setGeometry(0,0,400,200)
        self.setWindowTitle('Camera Setting')
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        acquisitionPanel = QGroupBox("Acquisition",self)
        main_layout.addWidget(acquisitionPanel)
        acquisition_layout = QGridLayout()
        acquisitionPanel.setLayout(acquisition_layout)
        # Create spin box for exposure value
        self.exposureSpinBox = TouchIntSpinBox(0)
        self.exposureSpinBox.valueChanged.connect(self.onExposureChanged)
        self.exposureSpinBox.setRange(*feature_value["ExposureTimeAbs"]['range'])
        self.exposureSpinBox.setSingleStep(1)
        self.exposureSpinBox.setValue(int(feature_value["ExposureTimeAbs"]['value']))
        # Create spin box for gain value
        self.gainSpinBox = TouchIntSpinBox(0)
        self.gainSpinBox.valueChanged.connect(self.onGainChanged)
        self.gainSpinBox.setRange(*feature_value["Gain"]['range'])
        self.gainSpinBox.setSingleStep(1)
        self.gainSpinBox.setValue(int(feature_value["Gain"]['value']))
        
        # Create labels for spin boxes
        self.exposureLabel = QLabel("Exposure:")
        self.gainLabel = QLabel("Gain:")

        # Create layout and add widgets
        acquisition_layout.addWidget(self.exposureLabel,0,0)
        acquisition_layout.addWidget(self.exposureSpinBox,0,1)

        acquisition_layout.addWidget(self.gainLabel,1,0)
        acquisition_layout.addWidget(self.gainSpinBox,1,1)


        trigger_panel = QGroupBox("Trigger", self)
        main_layout.addWidget(trigger_panel)
        trigger_layout = QGridLayout()
        trigger_panel.setLayout(trigger_layout)

        self.trigger_source = QtWidgets.QComboBox()
        self.trigger_source.addItems(trigger_source_list)
        self.trigger_source.setCurrentIndex(trigger_source_list.index(str(feature_value["TriggerSource"])))
        self.trigger_source.currentTextChanged.connect(self.trigger_source_currentTextChanged)
        

        self.trigger_activation = QtWidgets.QComboBox()
        self.trigger_activation.addItems(trigger_activation_list)
        self.trigger_activation.setCurrentIndex(trigger_activation_list.index(str(feature_value["TriggerActivation"])))
        self.trigger_activation.currentTextChanged.connect(self.trigger_activation_currentTextChanged)
        

        self.trigger_selector = QtWidgets.QComboBox()
        self.trigger_selector.addItems(trigger_selector_list)
        self.trigger_selector.setCurrentIndex(trigger_selector_list.index(str(feature_value["TriggerSelector"])))
        self.trigger_selector.currentTextChanged.connect(self.trigger_selector_currentTextChanged)
        

        self.trigger_mode = QtWidgets.QComboBox()
        self.trigger_mode.addItems(trigger_mode_list)
        self.trigger_mode.setCurrentIndex(trigger_mode_list.index(str(feature_value["TriggerMode"])))
        self.trigger_mode.currentTextChanged.connect(self.trigger_mode_currentTextChanged)
        

        trigger_layout.addWidget(QLabel('Source:'),0,0)
        trigger_layout.addWidget(self.trigger_source,1,0)
        trigger_layout.addWidget(QLabel('Activation:'),0,1)
        trigger_layout.addWidget(self.trigger_activation,1,1)
        trigger_layout.addWidget(QLabel('Selector:'),2,0)
        trigger_layout.addWidget(self.trigger_selector,3,0)
        trigger_layout.addWidget(QLabel('Mode:'),2,1)
        trigger_layout.addWidget(self.trigger_mode,3,1)


        user_set_panel = QGroupBox("SavedUserSet",self)
        main_layout.addWidget(user_set_panel)
        user_set_layout = QGridLayout()
        user_set_panel.setLayout(user_set_layout)

        self.user_default_selector = QtWidgets.QComboBox()
        self.user_default_selector.addItems(user_default_selector_list)
        self.user_default_selector.setCurrentIndex(user_default_selector_list.index(str(feature_value["UserSetDefaultSelector"])))
        self.user_default_selector.currentTextChanged.connect(self.user_default_selector_currentTextChanged)
        

        self.user_selector = QtWidgets.QComboBox()
        self.user_selector.addItems(user_selector_list)
        self.user_selector.setCurrentIndex(user_selector_list.index(str(feature_value["UserSetSelector"])))
        self.user_selector.currentTextChanged.connect(self.user_selector_currentTextChanged)
        

        self.user_set_load = QtWidgets.QPushButton("Load")
        self.user_set_load.clicked.connect(self.user_set_load_clicked)

        self.user_set_save = QtWidgets.QPushButton("Save")
        self.user_set_save.clicked.connect(self.user_set_save_clicked)

        user_set_layout.addWidget(QLabel('UserSetDefaultSelector:'),0,0)
        user_set_layout.addWidget(self.user_default_selector,1,0)
        user_set_layout.addWidget(QLabel('Load:'),0,1)
        user_set_layout.addWidget(self.user_set_load,1,1)
        user_set_layout.addWidget(QLabel('UserSetSelector:'),2,0)
        user_set_layout.addWidget(self.user_selector,3,0)
        user_set_layout.addWidget(QLabel('Save:'),2,1)
        user_set_layout.addWidget(self.user_set_save,3,1)


    def trigger_source_currentTextChanged(self):
        print(self.trigger_source.currentText())
        try:
            set_feature_value("TriggerSource",self.trigger_source.currentText())
        except:
            pass

    def trigger_activation_currentTextChanged(self):
        print(self.trigger_activation.currentText())
        try:
            set_feature_value("TriggerActivation",self.trigger_activation.currentText())
        except:
            pass
    
    def trigger_selector_currentTextChanged(self):
        print(self.trigger_selector.currentText())
        try:
            set_feature_value("TriggerSelector",self.trigger_selector.currentText())
        except:
            pass

    def trigger_mode_currentTextChanged(self):
        print(self.trigger_mode.currentText())
        try:
            set_feature_value("TriggerMode",self.trigger_mode.currentText())
        except:
            pass

    def user_selector_currentTextChanged(self):
        print(self.user_selector.currentText())
        try:
            set_feature_value("UserSetSelector",self.user_selector.currentText())
        except:
            pass

    def user_default_selector_currentTextChanged(self):
        print(self.user_default_selector.currentText())
        try:
            set_feature_value("UserSetDefaultSelector",self.user_default_selector.currentText())
        except:
            pass

    def user_set_load_clicked(self):
        print('load clicked')
        try:
            set_feature_value("UserSetLoad",True)
        except:
            pass

    def user_set_save_clicked(self):
        print('save clicked')
        try:
            set_feature_value("UserSetSave",True)
        except:
            pass

    @QtCore.pyqtSlot(int)
    def onExposureChanged(self,value):
        try:
            set_feature_value("ExposureTimeAbs",value)
            print('exposure: ',value)
        except:
            pass

    @QtCore.pyqtSlot(int)
    def onGainChanged(self,value):
        try:
            set_feature_value("Gain",value)
            print('gain: ',value)
        except:
            pass
from vmbpy import *
def get_feature_value(name):
    value={}
    with VmbSystem.get_instance() as vimba:
        with vimba.get_all_cameras()[0] as cam:
            for n in name:
                feature = cam.get_feature_by_name(n)
                feature_type = feature.get_type()
                if feature_type is BoolFeature:
                    value[n]=feature.get()
                elif feature_type is FloatFeature or feature_type is IntFeature:
                    value[n]={}
                    value[n]['value']=feature.get()
                    value[n]['range'] = feature.get_range()
                elif feature_type is EnumFeature or feature_type is StringFeature:
                    value[n]= feature.get().as_tuple()[0]
            return value
            
def set_feature_value(name,value):
    with VmbSystem.get_instance() as vimba:
        with vimba.get_all_cameras()[0] as cam:
            try:
                feature = cam.get_feature_by_name(name)
                f_type = feature.get_type()
                if f_type is CommandFeature:
                    feature.run()
                else:
                    feature.set(value)
                return True
            except:
                return False