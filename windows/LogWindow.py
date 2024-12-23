from datetime import datetime, timedelta
import os
import sys
import time
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,QLabel,QHeaderView,QLineEdit,QDateEdit,QGridLayout,QMessageBox
from PyQt5 import QtCore
import sqlite3

from components.init_database import DbContext
from windows.image_window import ImageWindow

def open_file(filename):
    try:
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
    except:
        print('Error: Cannot open file.')

class LogWindow(QWidget):
    def __init__(self, db_name):
        super().__init__()

        self.db_name = db_name
        self.page = 0
        self.rows_per_page = 20
        self.image_window = None
        self.initUI()

    def initUI(self):
        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['ID',"Result","Defects",'Image','Captured Date'])
        self.table.setColumnWidth(0,150)
        self.table.setColumnWidth(1,200)
        self.table.setColumnWidth(2,400)
        self.table.setColumnWidth(3,200)
        self.table.setColumnWidth(4,400)
        #Table will fit the screen horizontally
        self.table.horizontalHeader().setStretchLastSection(True)
        # self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        self.next_button = QPushButton('Next', self)
        self.prev_button = QPushButton('Previous', self)
        self.next_button.clicked.connect(self.next_page)
        self.prev_button.clicked.connect(self.prev_page)
        self.lb_page = QLabel()

        self.search_input = QLineEdit(self)
        self.search_defect_input = QLineEdit(self)
        self.from_date = QDateEdit(datetime.now()-timedelta(days=1),self)
        self.from_date.setCalendarPopup(True)
        self.to_date = QDateEdit(datetime.now(),self)
        self.to_date.setCalendarPopup(True)
        self.search_button = QPushButton('Search', self)
        self.search_button.clicked.connect(self.search_button_clicked)

        
        
        nav_panel_layout = QHBoxLayout()
        nav_panel_layout.addWidget(self.prev_button)
        nav_panel_layout.addWidget(self.lb_page,alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        nav_panel_layout.addWidget(self.next_button)
        nav_panel = QWidget()
        nav_panel.setLayout(nav_panel_layout)
        
        search_panel_layout = QHBoxLayout()
        
        search_panel_layout.addWidget(QLabel("From:"))
        search_panel_layout.addWidget(self.from_date)
        search_panel_layout.addWidget(QLabel("To:"))
        search_panel_layout.addWidget(self.to_date)

        search_panel_layout.addWidget(QLabel("Search by Result:"))
        search_panel_layout.addWidget(self.search_input)
        search_panel_layout.addWidget(QLabel("Search by Defects:"))
        search_panel_layout.addWidget(self.search_defect_input)
        search_panel_layout.addWidget(self.search_button)

        search_panel = QWidget()
        search_panel.setLayout(search_panel_layout)

        vbox = QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(search_panel)
        vbox.addWidget(self.table)
        vbox.addWidget(nav_panel)

        self.setWindowTitle('Log Window')
        self.resize(600, 500)
        self.search_data()

    def search_button_clicked(self):
        self.page = 0
        self.search_data()

    def search_data(self):
        name = self.search_input.text()
        defect = self.search_defect_input.text()
        from_date = self.from_date.date().toString("yyyy-MM-dd")
        to_date = self.to_date.date().toString("yyyy-MM-dd")
        with DbContext(self.db_name) as db:
            st = time.time()
            sql = "SELECT COUNT(*) FROM tasks WHERE id_number LIKE ? AND defect_type LIKE ? AND captured_date BETWEEN ? AND ? ORDER BY id DESC"
            count = db.execute(sql,('%' + name + '%','%' + defect + '%', from_date, to_date)).fetchone()[0]
            data = db.fetchall(f'SELECT id, id_number, defect_type, image_path, captured_date FROM tasks WHERE id_number LIKE ? AND defect_type LIKE ? AND captured_date BETWEEN ? AND ? ORDER BY id DESC LIMIT {self.rows_per_page} OFFSET ?', ('%' + name + '%','%' + defect + '%', from_date, to_date, self.page * self.rows_per_page,))
            print('fetch log:',time.time()-st)
            self.total_page = int(count/self.rows_per_page)+1
            self.lb_page.setText('{}/{}'.format(str(self.page+1),str(self.total_page)))
            self.table.setRowCount(len(data))
            self.table.clearContents()
            for i, row in enumerate(data):
                for j, value in enumerate(row):
                    if j==3:
                        if(value is not None):
                            btn_open = QPushButton('Open')
                            btn_open.setToolTip(value)
                            btn_open.clicked.connect(self.showImage)
                            self.table.setCellWidget(i,j,btn_open)
                            # self.table.cellWidget(i,j).setToolTip(value)
                            # self.table.cellWidget(i,j).clicked.connect(lambda:self.showImage(self.table.cellWidget(i,j).toolTip()))
                            continue
                        else:
                            item = QTableWidgetItem("Not Record")
                            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                            self.table.setItem(i, j, item)
                            continue
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(i, j, item)
            print('reload log:',time.time()-st)
            
    def showImage(self):
        print(self.sender().toolTip())
        image_path = self.sender().toolTip()
        if not os.path.exists(image_path):
            QMessageBox.about(self,"Error", "Cannot open image. Image might be removed")
            return
        #open_file(image_path)
        self.open_image(image_path)
    def open_image(self,image_path):
        if(not self.image_window):
            self.image_window = ImageWindow(image_path)
            self.image_window.show()
        else:
            if (self.image_window.isVisible()):
                self.image_window.change_image(image_path)
            else:
                self.image_window = ImageWindow(image_path)
                self.image_window.show()
    def next_page(self):
        if self.page<self.total_page-1:
            self.page += 1
            self.search_data()

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.search_data()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)

#     log_window = LogWindow('yourdatabase.db')
#     log_window.show()

#     sys.exit(app.exec_())
