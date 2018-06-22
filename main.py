import sys
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(10, 10, 1200, 600)

        self.pushButton0 = QPushButton("Manual")
        self.pushButton0.setFixedHeight(600)
        self.pushButton0.clicked.connect(self.pushButton0Clicked)

        self.pushButton1= QPushButton("Training Data")
        self.pushButton1.setFixedHeight(600)
        self.pushButton1.clicked.connect(self.pushButton1Clicked)

        self.pushButton2= QPushButton("Predict")
        self.pushButton2.setFixedHeight(300)
        self.pushButton2.clicked.connect(self.pushButton2Clicked)

        self.pushButton3 = QPushButton("Administor")
        self.pushButton3.setFixedHeight(300)
        self.pushButton3.clicked.connect(self.pushButton3Clicked)


        self.Grid = QGridLayout()
        self.Grid.addWidget(self.pushButton0,0,0,2,1)
        self.Grid.addWidget(self.pushButton1,0,1,2,1)
        self.Grid.addWidget(self.pushButton2,0,2)
        self.Grid.addWidget(self.pushButton3,1,2)
        self.setLayout(self.Grid)
        #self.setLayout(self.rightlayout)
    def pushButton0Clicked(self):
        pass

    def pushButton1Clicked(self):
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        scikit = "/Users/uchan/Projects/keras_talk_py3/FTMP/scikit.py"
        subprocess.Popen([python_bin, scikit])
    def pushButton2Clicked(self):
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        keras = "/Users/uchan/Projects/keras_talk_py3/FTMP/test_csvView.py"
        subprocess.Popen([python_bin, keras])
    def pushButton3Clicked(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.setWindowTitle("UChanable")
    mywindow.show()
    app.exec_()