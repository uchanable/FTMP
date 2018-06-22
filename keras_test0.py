import sys
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class scikitDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUI()
    def setupUI(self):
        self.setGeometry(50,80,800,600)
        self.setWindowTitle("Scikit-Learn")

class kerasDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUI()
    def setupUI(self):
        self.setGeometry(50,80,800,600)
        self.setWindowTitle("Keras")

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(50, 80, 800, 600)

        self.pushButton1= QPushButton("Scikit-Learn")
        self.pushButton1.setFixedHeight(600)
        self.pushButton1.clicked.connect(self.pushButton1Clicked)

        self.pushButton2= QPushButton("Keras")
        self.pushButton2.setFixedHeight(600)
        self.pushButton2.clicked.connect(self.pushButton2Clicked)

        layout = QHBoxLayout()
        layout.addWidget(self.pushButton1)
        layout.addWidget(self.pushButton2)

        self.setLayout(layout)

    def pushButton1Clicked(self):
        dlg1 = scikitDialog()
        dlg1.exec_()
    def pushButton2Clicked(self):
        dlg2 = kerasDialog()
        dlg2.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.setWindowTitle("Keras")
    mywindow.show()
    app.exec_()