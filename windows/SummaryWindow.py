import sys
import os
from PyQt5.QtWidgets import QTableWidget,QLabel,QLineEdit,QTableWidgetItem, QApplication,QFrame,QSpacerItem,QCheckBox,QSpinBox,QLCDNumber,QListWidget, QMainWindow,QListWidgetItem, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QScrollArea, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPixmap,QFont,QPalette,QBrush,QColor
from PyQt5.QtCore import Qt,QRect,QCoreApplication,QTimer
from PyQt5 import QtCore

from components.Capture import ImageProcess
from components.custom_spinbox import TouchDoubleSpinBox
from tools.InstanceSegmentation import InstanceSegmentation
from tools.ONNXRuntime import ModelState
DefectMap1=[{'id': 0, 'name': 'bot-bac', 'supercategory': 'none'},
{'id': 1, 'name': 'dom-den', 'supercategory': 'none'},
{'id': 2, 'name': 'bot-bac-dinh-keo', 'supercategory': 'none'},
{'id': 3, 'name': 'vang-keo', 'supercategory': 'none'},
{'id': 4, 'name': 'nut', 'supercategory': 'none'},
{'id': 5, 'name': 'keo-arona', 'supercategory': 'none'},
{'id': 6, 'name': 'loi-day', 'supercategory': 'none'},
{'id': 7, 'name': 'xuoc', 'supercategory': 'none'}]
class SummaryWindow(QWidget):
    def __init__(self,image_process1:ImageProcess,image_process2:ImageProcess):
        super().__init__()
        self.image_process1= image_process1
        self.image_process2 = image_process2
        self.initUI()
        self.setWindowFlags(QtCore.Qt.Window  | Qt.WindowStaysOnTopHint)
    def initUI(self):
        vbox = QHBoxLayout()
        

        self.CamView1 = SummaryView("Camera 1",self.image_process1.Detection)
        self.CamView2 = SummaryView("Camera 2",self.image_process2.Detection)

        vbox.addWidget(self.CamView1,stretch=1)
        vbox.addWidget(self.CamView2,stretch=1)

        self.setLayout(vbox)
        self.resize(1400,800)

class SummaryView(QWidget):
    def __init__(self,title="Camera xxx",Detection : InstanceSegmentation = None):
        super().__init__()
        self.title = title
        self.Detection = Detection
        if Detection.State != ModelState.Loaded:
            Detection.LoadRecipe()
        if Detection is None:
            self.defect_map = DefectMap1
        else:
            self.defect_map = Detection.classes_map
        self.initUI()
        self.UpdateValue()
        self.checkThreadTimer = QTimer(self)
        self.checkThreadTimer.setSingleShot(False)
        self.checkThreadTimer.setInterval(500) #.5 seconds
        self.checkThreadTimer.timeout.connect(self.UpdateValue)
        self.checkThreadTimer.start()

    def initUI(self):
        
        self.tableWidget =  QTableWidget(len(self.defect_map),5, self)
        self.tableWidget.setHorizontalHeaderLabels(['Name','Count','Percent','Score','Min Size'])
        titleBox = QHBoxLayout()
        self.TitleLabel =  QLabel(self.title)
        titleBox.addWidget(self.TitleLabel)
        self.saveBtn = QPushButton("Save")
        self.saveBtn.clicked.connect(self.Save)
        titleBox.addWidget(self.TitleLabel)
        titleBox.addWidget(self.saveBtn)
        self.SummayLabel =  QLabel("Detail")
        self.DetailLabel =  QLabel("Detail")
        self.totalBox = QHBoxLayout()
        self.totalLabel = QLabel("0")    
        totalText = QLabel("Total")
        self.totalBox.addWidget(totalText, stretch=1)
        self.totalBox.addWidget(self.totalLabel, stretch=1)

        self.passBox = QHBoxLayout()
        self.passLabel = QLabel("0")
        okText = QLabel("OK")
        okText.setStyleSheet('QLabel {color : green; }')
        self.passBox.addWidget(okText, stretch=1)
        self.passBox.addWidget(self.passLabel, stretch=1)

        self.failBox = QHBoxLayout()
        self.failLabel = QLabel("0")
        ngText = QLabel("NG")
        ngText.setStyleSheet('QLabel {color : red; }')
        self.failBox.addWidget(ngText, stretch=1)
        self.failBox.addWidget(self.failLabel, stretch=1)

        self.currentPercentBox = QHBoxLayout()
        self.currentPercentBox.addWidget(QLabel("Current"), stretch=1)
        self.currentPercentLabel = QLabel("0%")
        self.currentPercentBox.addWidget(self.currentPercentLabel, stretch=1)

        self.thresholdBox = QHBoxLayout()
        self.thresholdBox.addWidget(QLabel("Threshold"), stretch=1)
        self.thresholdBox.addWidget(QLineEdit("50"), stretch=1)

        self.statusBox = QHBoxLayout()
        self.statusBox.addWidget(QLabel("Status"), stretch=1)
        self.statusBox.addWidget(QLabel("OK"), stretch=1)

        self.bypassBox = QHBoxLayout()
        self.bypassBox.addWidget(QLabel("Bypass"), stretch=1)
        self.bypassCheckBox = QCheckBox()
        self.bypassCheckBox.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "width :30px;"
                               "height : 30px;"
                               "}")
        self.bypassBox.addWidget(self.bypassCheckBox, stretch=1)

        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.resetCounter)
        vbox = QVBoxLayout()
        vbox.addItem(titleBox)
        vbox.addItem(self.totalBox)
        vbox.addItem(self.passBox)
        vbox.addItem(self.failBox)
        vbox.addWidget(self.resetButton)
        vbox.addWidget(self.tableWidget)
        vbox.addWidget(QLabel("Setting"))
        vbox.addItem(self.currentPercentBox)
        vbox.addItem(self.thresholdBox)
        vbox.addItem(self.statusBox)
        vbox.addItem(self.bypassBox)
        self.setLayout(vbox)

        
        self.ValueLabels = []
        self.PercentLabels = []
        self.thresholdValues = []
        self.minSizeValues = []
        row=0
        for data in self.defect_map:
            label = QLabel(data['name'])
            valueLabel = QLabel(str(data['count']))
            valueLabel.setAlignment(Qt.AlignRight)

            percentLabel = QLabel('0%')
            percentLabel.setAlignment(Qt.AlignRight)

            thresholdInput = TouchDoubleSpinBox()
            thresholdInput.setValue(data['score_threshold'])
            thresholdInput.valueChanged.connect(lambda value, index=row: self.thresholdChanged(value,index))
            #thresholdInput.setAlignment(Qt.AlignRight)

            minSizeInput = TouchDoubleSpinBox()
            minSizeInput.setValue(data['min_size'])
            minSizeInput.valueChanged.connect(lambda value, index=row: self.minSizeChanged(value,index))
            #minSizeInput.setAlignment(Qt.AlignRight)

            self.ValueLabels.append(valueLabel)
            self.PercentLabels.append(percentLabel)
            self.thresholdValues.append(thresholdInput)
            self.minSizeValues.append(minSizeInput)
            self.tableWidget.setCellWidget(row,0,label)
            self.tableWidget.setCellWidget(row,1,valueLabel)
            self.tableWidget.setCellWidget(row,2,percentLabel)
            self.tableWidget.setCellWidget(row,3,thresholdInput)
            self.tableWidget.setCellWidget(row,4,minSizeInput)
            self.tableWidget.setRowHeight(row,50)            
            row=row+1
        self.tableWidget.setColumnWidth(3,150)
        self.tableWidget.setColumnWidth(4,150)
        return
    def Save(self):
        self.Detection.save()
    def thresholdChanged(self,value,index):
        self.Detection.classes_map[index]['score_threshold'] = value
    def minSizeChanged(self,value,index):
        self.Detection.classes_map[index]['min_size'] = value
    def resetCounter(self):
        self.Detection.total_count=0
        self.Detection.pass_count=0
        self.Detection.fail_count=0
        for item in self.Detection.classes_map:
            item['count']=0
    def UpdateValue(self):
        self.totalLabel.setText(str(self.Detection.total_count))
        self.passLabel.setText(str(self.Detection.pass_count))
        self.failLabel.setText(str(self.Detection.fail_count))
        if(self.Detection.total_count>0):
            self.currentPercentLabel.setText(f'{self.Detection.fail_count*100.0/self.Detection.total_count:.2f}%')
        else:
            self.currentPercentLabel.setText(f'0%')
        for i in range(len(self.defect_map)):
            self.ValueLabels[i].setText(str(self.defect_map[i]['count']))
            if(self.Detection.total_count>0):
                self.PercentLabels[i].setText(f'{self.defect_map[i]["count"]*100.0/self.Detection.total_count:.2f}%')
            else:
                self.PercentLabels[i].setText(f'0%')
