import json
import os
import shutil
from PyQt5.QtWidgets import QWidget,QGridLayout,QGroupBox,QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox,QLabel,QFileDialog,QSpacerItem,QSizePolicy
from components.custom_spinbox import TouchDoubleSpinBox, TouchIntSpinBox
from PyQt5 import QtCore
from tools.InstanceSegmentation import InstanceSegmentation
from windows.CreateRegionWindow import CreateRegionWindow
from windows.ToolSettingView import ToolSetting

class InstanceSegmentationView(QWidget,ToolSetting):
    def __init__(self, tool:InstanceSegmentation, parent=None):
        super(InstanceSegmentationView, self).__init__(parent)
        self.tool = tool
        self.initUI()
    def initUI(self):
        self.layout = QVBoxLayout()
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # Load model directory control
        #load_model_group = QGroupBox("Load model",self)
        self.load_model_button = QPushButton('Choose model file', self)
        self.load_model_button.clicked.connect(self.load_model_clicked)
        # Change region directory control
        #change_region_group = QGroupBox("Change inspection region",self)
        self.change_region_button = QPushButton('Change Region', self)
        self.change_region_button.clicked.connect(self.change_region_clicked)

        self.layout.addWidget(self.load_model_button)
        self.layout.addWidget(self.change_region_button)
        self.layout.addItem(verticalSpacer)
        self.setLayout(self.layout)
        
    def change_region_clicked(self):
        self.w = CreateRegionWindow(self.tool)
        self.w.show()
    def load_model_clicked(self):
        file_name,_ = QFileDialog.getOpenFileName(self, "Select file")
        if file_name:  
            try:
                shutil.copyfile(file_name,os.path.join(self.tool.ModelDir,'model_det.onnx'))
                self.tool.LoadModel(file_name)
            except: 
                QMessageBox.about(self,"Error","Cannot load model")
            
    @QtCore.pyqtSlot(float)
    def onThreshChanged(self,value):
        self.tool.postprocess_params = value
        self.tool.thresh = value

    @QtCore.pyqtSlot(float)
    def onBoxThreshChanged(self,value):
        self.tool.postprocess_op.box_thresh = value
        self.tool.box_thresh = value

    @QtCore.pyqtSlot(int)
    def onMaxCandidatesChanged(self,value):
        self.tool.postprocess_op.max_candidates = value
        self.tool.max_candidates = value

    @QtCore.pyqtSlot(float)
    def onUnclipRatioChanged(self,value):
        self.tool.postprocess_op.unclip_ratio = value
        self.tool.unclip_ratio = value

    @QtCore.pyqtSlot(int)
    def onMinSizeChanged(self,value):
        self.tool.postprocess_op.min_size = value
        self.tool.min_size = value