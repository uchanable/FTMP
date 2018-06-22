import sys
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(50, 80, 800, 600)

        self.pushButton1 = QPushButton("Iris Plants Database")
        self.pushButton2 = QPushButton("Predict House Prices in Boston")
        self.pushButton3 = QPushButton("Custom Classfication")
        self.pushButton4 = QPushButton("Custom Regression")
        self.pushButton5 = QPushButton("5")
        self.pushButton6 = QPushButton("6")
        self.pushButton7 = QPushButton("7")
        self.pushButton8 = QPushButton("8")

        self.pushButton1.setFixedHeight(300)
        self.pushButton2.setFixedHeight(300)
        self.pushButton3.setFixedHeight(300)
        self.pushButton4.setFixedHeight(300)
        #self.pushButton5.setFixedHeight(300)
        #self.pushButton6.setFixedHeight(300)
        #self.pushButton7.setFixedHeight(300)
        #self.pushButton8.setFixedHeight(300)

        self.pushButton1.clicked.connect(self.pushButton1Clicked)
        self.pushButton2.clicked.connect(self.pushButton2Clicked)
        self.pushButton3.clicked.connect(self.pushButton3Clicked)
        self.pushButton4.clicked.connect(self.pushButton4Clicked)
        #self.pushButton5.clicked.connect(self.pushButton5Clicked)
        #self.pushButton6.clicked.connect(self.pushButton6Clicked)
        #self.pushButton7.clicked.connect(self.pushButton7Clicked)
        #self.pushButton8.clicked.connect(self.pushButton8Clicked)

        layout = QGridLayout()
        layout.addWidget(self.pushButton1,0,0)
        layout.addWidget(self.pushButton2,0,1)
        layout.addWidget(self.pushButton3,1,0)
        layout.addWidget(self.pushButton4,1,1)
        #layout.addWidget(self.pushButton5,1,0)
        #layout.addWidget(self.pushButton6,1,1)
        #layout.addWidget(self.pushButton7,1,2)
        #layout.addWidget(self.pushButton8,1,3)


        self.setLayout(layout)

    def pushButton1Clicked(self):
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        scikit = "/Users/uchan/Projects/keras_talk_py3/FTMP/scikit_ex/classfication01.py"
        subprocess.Popen([python_bin, scikit])
    def pushButton2Clicked(self):
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        scikit = "/Users/uchan/Projects/keras_talk_py3/FTMP/scikit_ex/regression01.py"
        subprocess.Popen([python_bin, scikit])
    def pushButton3Clicked(self):
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        scikit = "/Users/uchan/Projects/keras_talk_py3/FTMP/scikit_ex/classfication02.py"
        subprocess.Popen([python_bin, scikit])
    def pushButton4Clicked(self):
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        scikit = "/Users/uchan/Projects/keras_talk_py3/FTMP/scikit_ex/regression02.py"
        subprocess.Popen([python_bin, scikit])
    '''
    def pushButton5Clicked(self):
        pass
    def pushButton6Clicked(self):
        pass
    def pushButton7Clicked(self):
        pass
    def pushButton8Clicked(self):
        pass'''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.setWindowTitle("Scikit-Learn")
    mywindow.show()
    app.exec_()