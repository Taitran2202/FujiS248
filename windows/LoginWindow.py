from PyQt5.QtWidgets import  QWidget, QPushButton, QVBoxLayout, QLineEdit, QMessageBox,QLabel
from PyQt5.QtCore import  pyqtSignal
from components.UserContext import UserContext
from PyQt5 import QtGui,QtCore

class LoginWindow(QWidget):
    logged_in = pyqtSignal()
    def __init__(self, user_context:UserContext):
        super().__init__()

        self.user_context = user_context
        self.initUI()

    def initUI(self):
        self.username_edit = QLineEdit(self)
        self.password_edit = QLineEdit(self)
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.btn = QPushButton('Login', self)
        self.btn.clicked.connect(self.login)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel('LOGIN'),1,alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(QLabel("Username"))
        vbox.addWidget(self.username_edit)
        vbox.addWidget(QLabel("Password"))
        vbox.addWidget(self.password_edit)
        vbox.addWidget(self.btn,2)

        self.setLayout(vbox)

        self.setWindowTitle('Login Window')
        self.setFixedWidth(300)

    def login(self):
        # Update the UserContext with the entered username and role
        if self.user_context.login(self.username_edit.text(),self.password_edit.text()):
            self.logged_in.emit()
            self.close()
            return
        QMessageBox.about(self, "Info", "Invalid username or password?")
        return
        
