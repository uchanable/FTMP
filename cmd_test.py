import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup()

    def setup(self):
        # create objects
        label = QLabel(self.tr("Enter command and press Return"))
        self.le = QLineEdit()
        self.te = QTextEdit()

        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.le)
        layout.addWidget(self.te)
        self.setLayout(layout)

        # create connection
        #self.connect(self.le, SIGNAL("returnPressed(void)"),self.run_command)
        #self.le.textChanged(self.run_command)
        self.le.returnPressed.connect(self.run_command)

    def run_command(self):
        cmd = str(self.le.text())
        stdouterr = os.popen(cmd).read()
        self.te.setText(stdouterr)
        self.le.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())