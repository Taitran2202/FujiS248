import os
import shutil
from PyQt5.QtWidgets import QWidget,QGroupBox,QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox,QFileDialog
from tools.TextRecognition import TextRecognition
from windows.ToolSettingView import ToolSetting

class TextRecognitionSettingView (QWidget,ToolSetting):
    def __init__(self, tool:TextRecognition, parent=None):
        super(TextRecognitionSettingView, self).__init__(parent)
        self.tool = tool
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Load model directory control
        load_model_group = QGroupBox("Load model",self)
        self.load_model_button = QPushButton('Choose model file', self)
        self.load_model_button.clicked.connect(self.load_model_clicked)

        load_model_layout = QHBoxLayout()
        load_model_layout.setSpacing(0)
        load_model_layout.addWidget(self.load_model_button)
        load_model_group.setLayout(load_model_layout)

        self.layout.addWidget(load_model_group)

        self.setLayout(self.layout)
    
    def load_model_clicked(self):
        file_name,_ = QFileDialog.getOpenFileName(self, "Select file")
        if file_name:  
            try:
                shutil.copyfile(file_name,os.path.join(self.tool.ModelDir,'model_rec.onnx'))
                self.tool.LoadRecipe()
            except: 
                QMessageBox.about(self,"Error","Cannot load model")
            
