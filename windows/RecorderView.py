from datetime import datetime
from PyQt5.QtWidgets import QWidget,QBoxLayout,QGroupBox,QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox,QLabel,QComboBox,QLineEdit,QFileDialog,QStackedLayout

from components.Recorder import Recorder
IMAGE_FORMAT_LIST = [".bmp", ".png", ".jpg"]
class RecorderView(QWidget):
    def __init__(self, recorder:Recorder, parent=None):
        super(RecorderView, self).__init__(parent)
        self.recorder = recorder
        self.initUI()

    def initUI(self):
        self.layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)

        self.record_check = QCheckBox("Enable Record", self)
        self.record_check.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "width :30px;"
                               "height : 30px;"
                               "}")
        self.record_check.setChecked(self.recorder.enable_record)
        self.record_check.stateChanged.connect(self.toggle_record)

        # Image format control
        format_label = QGroupBox("Image Format", self)
        format_layout = QHBoxLayout()
        format_label.setLayout(format_layout)
        self.format_combo = QComboBox(self)
        self.format_combo.addItems(IMAGE_FORMAT_LIST)
        self.format_combo.setCurrentIndex(IMAGE_FORMAT_LIST.index(self.recorder.image_format))
        self.format_combo.currentTextChanged.connect(self.change_format)
        format_layout.addWidget(self.format_combo)

        # Save directory control
        save_dir_group = QGroupBox("Save Directory",self)
        self.save_dir_edit = QLineEdit(self)
        self.save_dir_edit.setReadOnly(True)
        self.save_dir_edit.setText(self.recorder.save_dir)
        self.save_dir_button = QPushButton('Change Directory', self)
        self.save_dir_button.clicked.connect(self.change_save_dir)

        save_dir_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        #save_dir_layout.setSpacing(0)
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(self.save_dir_button)
        save_dir_group.setLayout(save_dir_layout)

        # Folder and image name format controls
        folder_name_format_label = QGroupBox("Folder Name Format", self)
        folder_name_layout = QVBoxLayout()
        folder_name_format_label.setLayout(folder_name_layout)
        self.folder_name_format_edit = QLineEdit(self)
        self.folder_name_format_edit.setText(self.recorder.folder_name_format)
        self.folder_name_format_edit.textChanged.connect(self.update_example_folder_name)
        example_date = datetime.now()
        self.folder_example_label = QLabel("Example: " + example_date.strftime(self.recorder.folder_name_format), self)
        folder_name_layout.addWidget(self.folder_name_format_edit)
        folder_name_layout.addWidget(self.folder_example_label)

        image_name_format_label = QGroupBox("Image Name Format:", self)
        image_name_layout = QVBoxLayout()
        image_name_format_label.setLayout(image_name_layout)
        self.image_name_format_edit = QLineEdit(self)
        self.image_name_format_edit.setText(self.recorder.image_name_format)
        self.image_name_format_edit.textChanged.connect(self.update_example_image_name)
        self.image_example_label = QLabel("Example: " + example_date.strftime(self.recorder.image_name_format), self)
        image_name_layout.addWidget(self.image_name_format_edit)
        image_name_layout.addWidget(self.image_example_label)

        self.layout.addWidget(self.record_check)
        self.layout.addWidget(format_label)
        # self.layout.addWidget(self.format_combo)
        self.layout.addWidget(save_dir_group)
        # self.layout.addWidget(self.save_dir_edit)
        # self.layout.addWidget(self.save_dir_button)
        self.layout.addWidget(folder_name_format_label)
        # self.layout.addWidget(self.folder_name_format_edit)
        # self.layout.addWidget(self.folder_example_label)
        self.layout.addWidget(image_name_format_label)
        # self.layout.addWidget(self.image_name_format_edit)
        # self.layout.addWidget(self.image_example_label)
        self.saveBtn = QPushButton("Save")
        self.saveBtn.clicked.connect(self.save)

        self.layout.addWidget(self.saveBtn)
        self.setLayout(self.layout)

        self.setWindowTitle('Recorder Setting')
        self.resize(400, 300)
    def save(self):
        self.recorder.save()
    def toggle_record(self, state):
        self.recorder.enable_record = bool(state)

    def change_format(self, text):
        self.recorder.image_format = text

    def change_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:  # If a directory was selected
            self.recorder.save_dir = dir_path
            self.save_dir_edit.setText(dir_path)  # Update the line edit

    def update_example_folder_name(self):
        try:
            example_date = datetime.now()
            folder_example = example_date.strftime(self.folder_name_format_edit.text())
            self.folder_example_label.setText("Example: " + folder_example)
            self.recorder.folder_name_format = self.folder_name_format_edit.text()
        except: pass

    def update_example_image_name(self):
        try:
            example_date = datetime.now()
            image_example = example_date.strftime(self.image_name_format_edit.text())
            self.image_example_label.setText("Example: " + image_example)
            self.recorder.image_name_format = self.image_name_format_edit.text()
        except: pass