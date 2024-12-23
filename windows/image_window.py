from PyQt5.QtWidgets import QApplication,QListWidgetItem,QAction, QWidget, \
    QVBoxLayout, QPushButton, QFileDialog, QLabel,QProgressBar,QListWidget, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QMainWindow
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.image_path = image_path

        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.create_toolbar()
        self.load_image(image_path)

        self.setWindowTitle('Image Viewer')
        self.setGeometry(300, 300, 800, 600)
    def load_image(self,image_path):
        pixmap = QPixmap(image_path)
        self.pixmap_item.setPixmap(pixmap)

    def change_image(self,image_path):
        self.load_image(image_path)

    def create_toolbar(self):
        zoom_in_action = QAction('Zoom In', self)
        zoom_out_action = QAction('Zoom Out', self)
        fit_to_window_action = QAction('Fit to Window', self)

        zoom_in_action.triggered.connect(self.zoom_in)
        zoom_out_action.triggered.connect(self.zoom_out)
        fit_to_window_action.triggered.connect(self.fit_to_window)

        toolbar = self.addToolBar('Image Toolbar')
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(fit_to_window_action)


    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1 / 1.2, 1 / 1.2)

    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)