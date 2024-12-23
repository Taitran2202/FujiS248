from PyQt5 import QtCore, QtGui, QtWidgets

class TouchIntSpinBox(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int)
    def __init__(self, value=0, *args, **kwargs):
        super(TouchIntSpinBox, self).__init__(*args, **kwargs)
        self.value = value
        self.min = 0
        self.max = 999
        self.step = 1
        # Set up buttons and label
        self.minusButton = QtWidgets.QPushButton('-', self)
        self.minusButton.clicked.connect(self.stepDown)
        self.plusButton = QtWidgets.QPushButton('+', self)
        self.plusButton.clicked.connect(self.stepUp)
        self.textbox_integervalidator = QtWidgets.QLineEdit(str(self.value))
        self.textbox_integervalidator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.textbox_integervalidator.textChanged.connect(self.textbox_textChanged)
        # self.textbox_integervalidator.setPlaceholderText("upto 3 digit value only accept")
        self.textbox_integervalidator.setValidator(QtGui.QIntValidator(self.min, self.max, self))
        # Set up layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.minusButton)
        layout.addWidget(self.textbox_integervalidator)
        layout.addWidget(self.plusButton)
    def textbox_textChanged(self):
        if(self.textbox_integervalidator.text()== ''):
            value = 0
        value = int(float(self.textbox_integervalidator.text()))
        self.value = value
        self.valueChanged.emit(value)
        
    def fitRange(self,value):
        value = min(self.max,value)
        value = max(self.min,value)
        return value
    
    def stepBy(self, steps):
        # Update value when buttons are clicked
        value = int(float(self.textbox_integervalidator.text())) + steps
        value = self.fitRange(value)
        self.value = value
        self.valueChanged.emit(value)
        self.textbox_integervalidator.setText(str(value))
    
    def stepUp(self):
        # Increment value when plus button is clicked
        self.stepBy(self.step)
    
    def stepDown(self):
        # Decrement value when minus button is clicked
        self.stepBy(-self.step)
    
    def setRange(self,min,max):
        self.min = int(min)
        self.max = int(max)
        self.textbox_integervalidator.setValidator(QtGui.QIntValidator(self.min, self.max, self))

    def setValue(self,value):
        self.value = value
        self.textbox_integervalidator.setText(str(value))

    def setSingleStep(self,step):
        self.step = step

    
class TouchDoubleSpinBox(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwargs):
        super(TouchDoubleSpinBox, self).__init__(*args, **kwargs)
        self.value = 0
        self.min = 0
        self.max = 999
        self.decimals = 1
        # Set up buttons and label
        self.minusButton = QtWidgets.QPushButton('-', self)
        self.minusButton.clicked.connect(self.stepDown)
        self.plusButton = QtWidgets.QPushButton('+', self)
        self.plusButton.clicked.connect(self.stepUp)
        self.textbox_integervalidator = QtWidgets.QLineEdit(str(self.value))
        self.textbox_integervalidator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.textbox_integervalidator.textChanged.connect(self.textbox_textChanged)
        # self.textbox_integervalidator.setPlaceholderText("upto 3 digit value only accept")
        self.textbox_integervalidator.setValidator(QtGui.QDoubleValidator(self.min, self.max,self.decimals, self))
        # Set up layout
        layout = QtWidgets.QHBoxLayout(self)
        
        layout.addWidget(self.minusButton)
        layout2 = QtWidgets.QStackedLayout(layout)
        layout2.addWidget(self.textbox_integervalidator)
        layout.addWidget(self.plusButton)
        self.setLayout(layout)
    def textbox_textChanged(self):
        if(self.textbox_integervalidator.text()== ''):
            value = 0
        else:            
            value = float(self.textbox_integervalidator.text())
        self.value = value
        self.valueChanged.emit(value)
        
    def fitRange(self,value):
        value = min(self.max,value)
        value = max(self.min,value)
        return value
    
    def stepBy(self, steps):
        # Update value when buttons are clicked
        value = round(float(self.textbox_integervalidator.text()) + steps,self.decimals)
        value = self.fitRange(value)
        self.value = value
        self.valueChanged.emit(value)
        self.textbox_integervalidator.setText(str(value))
    
    def stepUp(self):
        # Increment value when plus button is clicked
        step = 10**(-self.decimals)
        self.stepBy(step)
    
    def stepDown(self):
        # Decrement value when minus button is clicked
        step = 10**(-self.decimals)
        self.stepBy(-step)
    
    def setRange(self,min,max,decimals):
        self.min = float(min)
        self.max = float(max)
        self.decimals = int(decimals)
        self.textbox_integervalidator.setValidator(QtGui.QDoubleValidator(self.min, self.max,self.decimals, self))

    def setValue(self,value):
        self.value = value
        self.textbox_integervalidator.setText(str(value))


    
