from PyQt5 import QtCore, QtGui, QtWidgets

class InteractiveRectangle(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(InteractiveRectangle, self).__init__(parent)
        self.setMouseTracking(True)
        self.pixmap_image = QtGui.QPixmap(r"D:\NewOcean\Projects\demo-AIM\Images-Stain_Detect\WIN_20230410_16_06_11_Pro.jpg")
        self.rect = QtCore.QRect(50, 50, 100, 100)
        self.handleSize = 10
        self.handleSpace = -4
        self.handlePressed = None
        self.mousePressPos = None
        self.mouseMovePos = None
        self.startGeometry = None
        # Set window background to image
        self.palette = QtGui.QPalette()
        self.palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(self.pixmap_image))
        self.setPalette(self.palette)
        
        # Set window size to image size
        self.resize(self.pixmap_image.width(), self.pixmap_image.height())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect, self.pixmap_image)
        painter.setBrush(QtGui.QColor(0, 0, 0, 0))
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        painter.drawRect(self.rect)

        for i, handle in enumerate(self.handles()):
            if i == self.handlePressed:
                painter.setBrush(QtGui.QColor(255, 0, 0))
            else:
                painter.setBrush(QtGui.QColor(0, 0, 255))
            painter.drawRect(handle)

    def mousePressEvent(self, event):
        self.mousePressPos = event.pos()
        self.mouseMovePos = event.pos()
        for i, handle in enumerate(self.handles()):
            if handle.contains(event.pos()):
                self.handlePressed = i
                break
        # Initialize startGeometry attribute
        self.startGeometry = self.rect.getRect()

    def mouseMoveEvent(self, event):
        if self.handlePressed is not None:
            self.interactiveResize(event.pos())
        elif self.rect.contains(event.pos()):
            self.interactiveMove(event.pos())
        else:
            QtWidgets.QWidget.mouseMoveEvent(self, event)
        self.mouseMovePos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.handlePressed is not None:
                if not self.rect.contains(event.pos()):
                    self.handlePressed = None
                    return
                if QtCore.QRect(*self.startGeometry).contains(event.pos()):
                    self.handlePressed = None
                    return
                if (event.pos() - self.mousePressPos).manhattanLength() < 3:
                    handle = QtCore.QRectF(*self.handles()[self.handlePressed])
                    button = QtWidgets.QPushButton('X', self)
                    button.clicked.connect(lambda: button.deleteLater())
                    button.setGeometry(handle.toRect())
                    button.show()
                else:
                    for btn in filter(lambda x: isinstance(x, QtWidgets.QPushButton), self.children()):
                        btn.deleteLater()
            else:
                for btn in filter(lambda x: isinstance(x, QtWidgets.QPushButton), self.children()):
                    btn.deleteLater()
            self.handlePressed = None
            QtWidgets.QWidget.mouseReleaseEvent(self, event)

    def interactiveMove(self, mousePos):
        diff = mousePos - self.mouseMovePos
        self.rect.translate(diff)

    def interactiveResize(self, mousePos):
        index = self.handlePressed
        fromX = min(mousePos.x(), self.startGeometry[2])
        fromY = min(mousePos.y(), self.startGeometry[3])
        toX = max(mousePos.x(), self.startGeometry[0])
        toY = max(mousePos.y(), self.startGeometry[1])
        
        if index in [0]:
            toY += fromY - toY
            fromY -= fromY - toY
            toX += fromX - toX
            fromX -= fromX - toX
        elif index in [1]:
            toY += fromY - toY
            fromY -= fromY - toY
            fromX += toX - fromX
            toX -= toX - fromX
        elif index in [2]:
            fromY += toY - fromY
            toY -= toY - fromY
            fromX += toX - fromX
            toX -= toX - fromX
        elif index in [3]:
            fromY += toY - fromY
            toY -= toY - fromY
            toX += fromX - toX
            fromX -= fromX - toX

        if index in [4]:
            toY += fromY - toY
            fromY -= fromY - toY
        elif index in [5]:
            fromX += toX - fromX
            toX -= toX - fromX

        if index in [6]:
            fromY += toY - fromY
            toY -= toY - fromY
        elif index in [7]:
            toX += fromX - toX
            fromX -= fromX - toX

        self.rect = QtCore.QRect(fromX, fromY, toX-fromX, toY-fromY)

    def handles(self):
        s = self.handleSize
        b = self.handleSpace
        x1, y1, x2, y2 = self.rect.getCoords()
        return [QtCore.QRectF(x1-b-s, y1-b-s, s, s),
                QtCore.QRectF(x2+b, y1-b-s, s, s),
                QtCore.QRectF(x2+b, y2+b, s, s),
                QtCore.QRectF(x1-b-s, y2+b, s, s),
                QtCore.QRectF(x1+(x2-x1)/2-s/2, y1-b-s, s, s),
                QtCore.QRectF(x2+b, y1+(y2-y1)/2-s/2, s, s),
                QtCore.QRectF(x1+(x2-x1)/2-s/2, y2+b, s, s),
                QtCore.QRectF(x1-b-s, y1+(y2-y1)/2-s/2, s, s)]

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = InteractiveRectangle()
    window.show()
    sys.exit(app.exec_())
