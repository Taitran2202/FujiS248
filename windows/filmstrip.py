import sys
import os
from PyQt5.QtWidgets import QApplication,QSpacerItem,QListWidget, QMainWindow,QListWidgetItem, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QScrollArea, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
class FilmstripWindow(QWidget):
    def __init__(self,openImage):
        super().__init__()
        self.openImage = openImage
        self.initUI()
        self.setWindowFlags(QtCore.Qt.Window |  Qt.WindowStaysOnTopHint)

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_folder = None
        #self.image_labels = []
        self.selected_image = None

        self.add_images_button = QPushButton("Select Directory")
        self.add_images_button.clicked.connect(self.open_image_dialog)
        self.layout.addWidget(self.add_images_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.image_strip = QListWidget()
        self.image_strip.itemSelectionChanged.connect(self.select_image)

        self.scroll_area.setWidget(self.image_strip)

    def open_image_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.image_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", options=options)

        if self.image_folder:
            self.load_images()

    def load_images(self):
        self.image_strip.clear()
        #self.image_labels = []

        if self.image_folder:
            images = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            for image in images:
                image_path = os.path.join(self.image_folder, image)
                #pixmap = self.load_and_resize_image(image_path)
                label = QListWidgetItem(image)
                #Elabel.setPixmap(pixmap)
                #label.setText(image)
                #label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                #label.setAlignment(Qt.AlignCenter)
                #label.mousePressEvent = lambda event, img=image_path,selectedLabel = label: self.select_image(img,selectedLabel)
                #self.image_labels.append(label)
                self.image_strip.addItem(label)

    def load_and_resize_image(self, image_path, max_width=100, max_height=100):
        img = QPixmap(image_path)
        if img.width() > max_width or img.height() > max_height:
            img = img.scaled(max_width, max_height, aspectRatioMode=Qt.KeepAspectRatio)
        return img

    def select_image(self):
        if(len(self.image_strip.selectedItems())>0):
            image_path= os.path.join(self.image_folder, self.image_strip.selectedItems()[0].text())
            print(image_path)
            self.openImage(image_path)