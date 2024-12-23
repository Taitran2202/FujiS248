from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QVBoxLayout, QGroupBox, QHBoxLayout,QPushButton

class OutputSelectionWidget(QWidget):
    def __init__(self,label='output',selection_range:list=[]):
        super().__init__()
        self.label = label
        self.selection_range = selection_range
        self.init_ui()

    def init_ui(self):
        # Create widgets
        label = QLabel(self.label)
        self.port_combobox = QComboBox()
        self.port_combobox.addItems(self.selection_range)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.port_combobox)
        self.setLayout(layout)

class OutputSetting(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Output Setting')
        # Create PortSelectionWidget
        port_selection_widget = OutputSelectionWidget('Port',['0','1'])
        output_selection_widget = OutputSelectionWidget('Output',[str(i) for i in range(8)])
        # Create a QGroupBox and add the PortSelectionWidget to it
        group_box = QGroupBox('OK output setting')
        group_layout = QHBoxLayout()        
        group_box.setLayout(group_layout)
        group_layout.addWidget(port_selection_widget)
        group_layout.addWidget(output_selection_widget)


        port_selection_widget_ng = OutputSelectionWidget('Port',['0','1'])
        output_selection_widget_ng = OutputSelectionWidget('Output',[str(i) for i in range(8)])
        group_box_ng = QGroupBox('NG output setting')
        group_layout_ng = QHBoxLayout()        
        group_box_ng.setLayout(group_layout_ng)
        group_layout_ng.addWidget(port_selection_widget_ng)
        group_layout_ng.addWidget(output_selection_widget_ng)

        btn_save = QPushButton('Save')
        btn_save.clicked.connect(self.save)
        # Set up the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box)
        main_layout.addWidget(group_box_ng)
        main_layout.addWidget(btn_save)
        self.setLayout(main_layout)
    def save(self):      
        print('saved')