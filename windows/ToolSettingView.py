from datetime import datetime
import json
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget,QBoxLayout,QGroupBox,QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox,QLabel,QComboBox,QLineEdit,QFileDialog

IMAGE_FORMAT_LIST = [".bmp", ".png", ".jpg"]
class ToolSettingView(QWidget):
    def __init__(self, parent=None):
        super(ToolSettingView, self).__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        btn_save=QPushButton("Save settings")
        btn_save.clicked.connect(self.btn_save_clicked)
        self.layout.addWidget(btn_save)
        self.tools = []
        self.save_path = 'data/settings.txt'
    def add_view(self,name,view):
        try:
            label = QLabel(name)
            label.setFont(QtGui.QFont('SansSerif',15,QtGui.QFont.Weight.Bold))
            self.layout.addWidget(label)
            self.layout.addWidget(view)
            self.tools.append(view)
        except: pass

    def btn_save_clicked(self):
        data = {}
        for tool in self.tools:
            data[str(type(tool))] = tool.Save()
        with open(self.save_path,'w') as f:
            json.dump(data,f,indent=2)
    def load_tools(self):
        if not os.path.exists(self.save_path):
            return
        with open(self.save_path,'r') as f:
            data = json.load(f)
            
            for tool in self.tools:
                tool.Load(data[str(type(tool))])
class ToolSetting:
    def Save(self):
        pass
    def Load(self,data):
        pass