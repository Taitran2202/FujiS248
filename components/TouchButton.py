from PyQt5 import QtWidgets,QtGui

class TouchButton(QtWidgets.QPushButton):
    def __init__(self, *args, **kwargs):
        QtWidgets.QPushButton.__init__(self, *args, **kwargs)
        self.setMinimumHeight(50)
        self.setMinimumWidth(50)
        # self.setFont(QtGui.QFont('SansSerif',20,QtGui.QFont.Weight.Medium))
    #     self.setStyleSheet(u"QPushButton {\n"
    # "   border: 2px solid gray;\n"
    # # "   border-radius: 10px;\n"
    # "   padding: 0 8px;\n"
    # # "   background-color: rgb(40, 42, 54);\n"
    # "   font: bold 50px;\n"
    # "}\n"
    # "\n"
    # # "QPushButton:pressed {\n"
    # # "   background-color:rgb(0, 255, 127);\n"
    # # "}\n"
    # # "QPushButton:hover:!pressed {\n"
    # # "   background-color:rgb(51, 54, 69)\n"
    # # "}\n"
    # "")