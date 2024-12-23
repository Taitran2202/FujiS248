import json
import os
import shutil
from PyQt5.QtWidgets import QWidget,QGridLayout,QGroupBox,QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox,QLabel,QFileDialog
from components.custom_spinbox import TouchDoubleSpinBox, TouchIntSpinBox
from PyQt5 import QtCore
from tools.TextDetectionV2 import TextDetectionV2
from windows.CreateRegionWindow import CreateRegionWindow
from windows.ToolSettingView import ToolSetting

class TextDetectionV2SettingView(QWidget,ToolSetting):
    def __init__(self, tool:TextDetectionV2, parent=None):
        super(TextDetectionV2SettingView, self).__init__(parent)
        self.tool = tool
        self.initUI()
    def Save(self):
        config = {
            'region':self.tool.inspection_regions,
            'thresh':self.tool.thresh,
            'box_thresh':self.tool.box_thresh,
            'max_candidates':self.tool.max_candidates,
            'unclip_ratio':self.tool.unclip_ratio,
            'min_size':self.tool.min_size      
                  }
        print('Saved:',type(self))
        return config
    def Load(self,data):
        try:
            self.tool.inspection_regions=data['region']
            self.sb_thresh.textbox_integervalidator.setText(str(data['thresh']))
            self.sb_box_thresh.textbox_integervalidator.setText(str(data['box_thresh']))
            self.sb_max_candidates.textbox_integervalidator.setText(str(data['max_candidates']))
            self.sb_unclip_ratio.textbox_integervalidator.setText(str(data['unclip_ratio']))
            self.sb_min_size.textbox_integervalidator.setText(str(data['min_size']))
            print('Loaded:',type(self))
        except:
            print('Load failed:',type(self))
    def initUI(self):
        self.layout = QVBoxLayout()

        # Load model directory control
        load_model_group = QGroupBox("Load model",self)
        self.load_model_button = QPushButton('Choose model file', self)
        self.load_model_button.clicked.connect(self.load_model_clicked)

        # Change region directory control
        change_region_group = QGroupBox("Change inspection region",self)
        self.change_region_button = QPushButton('Change', self)
        self.change_region_button.clicked.connect(self.change_region_clicked)
        
        # parameters control
        parameterPanel = QGroupBox("Parameters",self)
        # parameterPanel.setMinimumHeight(500)
        # self.layout.addWidget(parameterPanel)
        parameter_layout = QGridLayout()
        parameterPanel.setLayout(parameter_layout)
        # Create spin box for thresh value
        self.sb_thresh = TouchDoubleSpinBox()
        self.sb_thresh.valueChanged.connect(self.onThreshChanged)
        self.sb_thresh.setRange(0,1,1)
        self.sb_thresh.setValue(self.tool.thresh)
        # Create spin box for box_thresh value
        self.sb_box_thresh = TouchDoubleSpinBox()
        self.sb_box_thresh.valueChanged.connect(self.onBoxThreshChanged)
        self.sb_box_thresh.setRange(0,1,1)
        self.sb_box_thresh.setValue(self.tool.box_thresh)
        # Create spin box for max_candidates value
        self.sb_max_candidates = TouchIntSpinBox(0)
        self.sb_max_candidates.valueChanged.connect(self.onMaxCandidatesChanged)
        self.sb_max_candidates.setRange(0,1000)
        self.sb_max_candidates.setValue(self.tool.max_candidates)
        # Create spin box for unclip_ratio value
        self.sb_unclip_ratio = TouchDoubleSpinBox()
        self.sb_unclip_ratio.valueChanged.connect(self.onUnclipRatioChanged)
        self.sb_unclip_ratio.setRange(0,10,1)
        self.sb_unclip_ratio.setValue(self.tool.unclip_ratio)
        # Create spin box for min_size value
        self.sb_min_size = TouchIntSpinBox(0)
        self.sb_min_size.valueChanged.connect(self.onMinSizeChanged)
        self.sb_min_size.setRange(0,9999999)
        self.sb_min_size.setValue(self.tool.min_size)
        
        # Create labels for spin boxes
        self.lb_thresh = QLabel("thresh:")
        self.lb_box_thresh = QLabel("box_thresh:")
        self.lb_max_candidates = QLabel("max_candidates:")
        self.lb_unclip_ratio = QLabel("unclip_ratio:")
        self.lb_min_size = QLabel("min_size:")

        # Create layout and add widgets
        parameter_layout.addWidget(self.lb_thresh,0,0)
        parameter_layout.addWidget(self.sb_thresh,0,1)

        parameter_layout.addWidget(self.lb_box_thresh,1,0)
        parameter_layout.addWidget(self.sb_box_thresh,1,1)

        parameter_layout.addWidget(self.lb_max_candidates,2,0)
        parameter_layout.addWidget(self.sb_max_candidates,2,1)

        parameter_layout.addWidget(self.lb_unclip_ratio,3,0)
        parameter_layout.addWidget(self.sb_unclip_ratio,3,1)
        
        parameter_layout.addWidget(self.lb_min_size,4,0)
        parameter_layout.addWidget(self.sb_min_size,4,1)

        load_model_layout = QHBoxLayout()
        load_model_layout.setSpacing(0)
        load_model_layout.addWidget(self.load_model_button)
        load_model_group.setLayout(load_model_layout)

        change_region_layout = QHBoxLayout()
        change_region_layout.setSpacing(0)
        change_region_layout.addWidget(self.change_region_button)
        change_region_group.setLayout(change_region_layout)

        self.layout.addWidget(load_model_group)
        self.layout.addWidget(change_region_group)
        self.layout.addWidget(parameterPanel)

        self.setLayout(self.layout)
    def change_region_clicked(self):
        self.w = CreateRegionWindow(self.tool)
        self.w.show()
    def load_model_clicked(self):
        file_name,_ = QFileDialog.getOpenFileName(self, "Select file")
        if file_name:  
            try:
                shutil.copyfile(file_name,os.path.join(self.tool.ModelDir,'model_det.onnx'))
                self.tool.LoadRecipe()
            except: 
                QMessageBox.about(self,"Error","Cannot load model")
            
    @QtCore.pyqtSlot(float)
    def onThreshChanged(self,value):
        self.tool.postprocess_op.thresh = value
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