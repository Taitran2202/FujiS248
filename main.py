from PyQt5 import QtWidgets, QtGui
import sys

from vmbpy import VmbSystem
from App import App
from components.init_database import init_database
from qt_material import apply_stylesheet
from PyQt5.QtCore import QFile, QTextStream
import breeze_resources
#import qdarktheme

if __name__ == "__main__":
    dbconn ='app.sqlite'
    init_database(dbconn)
    with VmbSystem.get_instance() as vmb:
        app = QtWidgets.QApplication(sys.argv)
        #app.setStyle("Fusion")
        extra = {
            
            'font_size': '14px',
            'density_scale': '0',
            'font_family': 'Roboto',
        }
        apply_stylesheet(app, theme='light_blue.xml',extra=extra)
        #qdarktheme.setup_theme()
        QtWidgets.QApplication.setFont(QtGui.QFont('Roboto',14,QtGui.QFont.Weight.Normal))
        # set stylesheet
        # file = QFile(":/light/stylesheet.qss")
        # file.open(QFile.ReadOnly | QFile.Text)
        # stream = QTextStream(file)
        # app.setStyleSheet(stream.readAll())
        ex = App(dbconn)
        ex.show()

        # window = MainWindow(dbconn)
        app.exec_()
        sys.exit(0)