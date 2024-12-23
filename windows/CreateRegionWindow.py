import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter,QImage,QCloseEvent
from PyQt5 import QtGui,QtCore

from tools.InstanceSegmentation import InstanceSegmentation


class CreateRegionWindow(QWidget):
	def __init__(self,tool:InstanceSegmentation):
		super().__init__()
		self.tool = tool
		self.window_width, self.window_height = 1200, 800
		self.setMinimumSize(self.window_width, self.window_height)

		layout = QVBoxLayout()
		self.setLayout(layout)

		self.pix = QPixmap(self.rect().size())
		if tool.image is not None:
			self.height, self.width, channel = tool.image.shape
			qimage = QtGui.QImage(tool.image.data, self.width, self.height, self.width*channel, QtGui.QImage.Format_RGB888)
		# image = QImage(image,)
			self.pix = self.pix.fromImage(QImage(qimage)).scaled(self.pix.size(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation)
		else:
			self.height=self.rect().height()
			self.width=self.rect().width()
			self.pix.fill(Qt.black)

		self.begin, self.destination = QPoint(), QPoint()	
		self.rects = []
    
	def paintEvent(self, event):
		painter = QPainter(self)
		pen = QtGui.QPen(QtGui.QColor(0,255,0))
		pen.setWidth(2)
		pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
		painter.setPen(pen)
		painter.drawPixmap(QPoint(), self.pix)

		if not self.begin.isNull() and not self.destination.isNull():
			
			rect = QRect(self.begin, self.destination)
			if rect.height()<10 or rect.width()<10:
				return
			painter.drawRect(rect.normalized())

	def mousePressEvent(self, event):
		if event.buttons() & Qt.LeftButton:
			print('Point 1')
			self.begin = event.pos()
			self.destination = self.begin
			self.update()

	def mouseMoveEvent(self, event):
		if event.buttons() & Qt.LeftButton:		
			print('Point 2')	
			self.destination = event.pos()
			self.update()

	def mouseReleaseEvent(self, event):
		print('Point 3')
		if event.button() & Qt.LeftButton:
			rect = QRect(self.begin, self.destination)
			if rect.height()<5 or rect.width()<5:
				return
			painter = QPainter(self.pix)
			pen = QtGui.QPen(QtGui.QColor(0,255,0))
			pen.setWidth(2)
			pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
			painter.setPen(pen)
			painter.drawRect(rect.normalized())
			self.rects.append(rect.getRect())
			self.begin, self.destination = QPoint(), QPoint()
			self.update()
	def get_rects(self):
		return [r.getRect() for r in self.rects]
	def closeEvent(self, a0: QCloseEvent) -> None:
		super().closeEvent(a0)
		y_scale = self.height / self.pix.height() 
		x_scale = self.width / self.pix.width()
		self.tool.inspection_regions = [[int(b[0]*x_scale),int(b[1]*y_scale),int(b[2]*x_scale),int(b[3]*y_scale)] for b in self.rects]
		print('close')
	    
