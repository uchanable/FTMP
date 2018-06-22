import sys
import pandas as pd
import random as rnd
import seaborn as sns
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

import autosklearn.classification
import sklearn.model_selection
import sklearn.metrics

import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sklearn.model_selection import train_test_split
from PandasModel import PandasModel
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pickle

iris_dataset = load_iris()
df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df["target"] = iris_dataset.target_names[iris_dataset.target]

class saveModel(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUI()
        self.name = None

    def setupUI(self):
        self.setGeometry(500, 200, 300, 100)
        self.setWindowTitle("Save AutoML Model")
        label1 = QLabel("Model Name : ")
        self.lineEdit = QLineEdit()
        self.savebtn = QPushButton("Save")
        self.savebtn.clicked.connect(self.savebtnClicked)
        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.lineEdit,0,1)
        layout.addWidget(self.savebtn,0,2)
        self.setLayout(layout)
    def savebtnClicked(self):
        self.name = self.lineEdit.text()
        self.close()
class ShowMatplotlib(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUI()
    def seabornplot(self):
        self.test = sns.pairplot(df, hue="target", size=2)
        return self.test.fig
    def setupUI(self):
        self.setGeometry(200, 200, 1000, 800)
        self.setWindowTitle("PyChart Viewer v0.1")
        self.lineEdit = QLineEdit()
        self.pushButton = QPushButton("Show Fig")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        self.fig = plt.Figure()
        self.fig1 = self.seabornplot()
        self.canvas = FigureCanvas(self.fig1)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.lineEdit)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)
    def pushButtonClicked(self):
        code = self.lineEdit.text()
        if code == "0":
            self.fig1.clear()
            self.fig1 = sns.violinplot(data=df, x="target", y ="sepal length (cm)",size=2)
            self.canvas.draw()

        elif code == "1":
            self.fig1.clear()
            self.fig1 = sns.violinplot(data=df, x="target", y ="sepal width (cm)",size=2)
            self.canvas.draw()

        elif code == "2":
            self.fig1.clear()
            self.fig1 = sns.violinplot(data=df, x="target", y ="petal length (cm)",size=2)
            self.canvas.draw()

        elif code == "3":
            self.fig1.clear()
            self.fig1 = sns.violinplot(data=df, x="target", y ="petal width (cm)",size=2)
            self.canvas.draw()
class TrainData(QDialog):
    def __init__(self):
        QDialog.__init__(self)
class PandasModelTrainData(QAbstractTableModel):
    def __init__(self, df1 = pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df1 = df1
        self._df1.sort_values(by='Score', ascending=False)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            try:
                return self._df1.columns.tolist()[section]
            except (IndexError, ):
                return QVariant()
        elif orientation == Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df1.index.tolist()[section]
            except (IndexError, ):
                return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if not index.isValid():
            return QVariant()

        return QVariant(str(self._df1.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df1.index[index.row()]
        col = self._df1.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df1[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df1.set_value(row, col, value)
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df1.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df1.columns)

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        self._df1.sort_values(by = "Score", ascending= order == Qt.AscendingOrder, inplace=True)
        self._df1.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()
class ShowDataFrame(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent=None)
        vLayout = QVBoxLayout(self)
        hLayout = QHBoxLayout()
        self.pathLE = QLineEdit(self)
        hLayout.addWidget(self.pathLE)
        self.loadBtn = QPushButton("Select File", self)
        hLayout.addWidget(self.loadBtn)
        vLayout.addLayout(hLayout)
        self.pandasTv = QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.loadBtn.clicked.connect(self.loadFile)
        self.pandasTv.setSortingEnabled(True)
        self.setGeometry(50,80,800,600)
        #model = pd.DataFrame(None)
        #self.pandasTv.setModel(model)
        self.pandasTv.setColumnWidth(0,200)

    def loadFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv)");
        self.pathLE.setText(fileName)
        df = pd.read_csv(fileName)
        model = PandasModel(df)
        self.pandasTv.setModel(model)

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
    def setupUI(self):
        self.setGeometry(50, 80, 800, 600)
        #self.browser = QTextEdit()
        self.browser = QTextEdit()
        self.b1 = QPushButton("Description")
        self.b2 = QPushButton("Load Dataset")
        self.b3 = QPushButton("Data Split (Train, Test Data)")
        self.b4 = QPushButton("Train Model")
        self.b5 = QPushButton("Auto Machine Learning")
        self.b6 = QPushButton("More Information about Data")
        self.b7 = QPushButton("Save AutoML Model")
        self.b8 = QPushButton("exit")

        self.b1.setFixedHeight(60)
        self.b2.setFixedHeight(60)
        self.b3.setFixedHeight(60)
        self.b4.setFixedHeight(60)
        self.b5.setFixedHeight(60)
        self.b6.setFixedHeight(60)
        self.b7.setFixedHeight(60)
        self.b8.setFixedHeight(60)

        self.GridLayout = QGridLayout()
        self.GridLayout.addWidget(self.b1,0,0)
        self.GridLayout.addWidget(self.b2,1,0)
        self.GridLayout.addWidget(self.b3,3,0)
        self.GridLayout.addWidget(self.b4,4,0)
        self.GridLayout.addWidget(self.b5,5,0)
        self.GridLayout.addWidget(self.b6,2,0)
        self.GridLayout.addWidget(self.b7,6,0)
        self.GridLayout.addWidget(self.b8,7,0)
        self.GridLayout.addWidget(self.browser,0,1,8,1)

        #self.b1.setEnabled(False)
        #self.b4.setEnabled(False)
        #self.b3.setEnabled(False)
        #self.b5.setEnabled(False)
        #self.b7.setEnabled(False)
        #self.b6.setEnabled(False)


        self.b1.clicked.connect(self.descrClicked)
        self.b2.clicked.connect(self.moreinfoClicked)
        self.b3.clicked.connect(self.datasplit)
        self.b4.clicked.connect(self.TrainModel)
        self.b5.clicked.connect(self.popup)
        self.b6.clicked.connect(self.showmatplot)
        self.b7.clicked.connect(self.pushb7clicked)
        self.b8.clicked.connect(QCoreApplication.instance().quit)


        self.setLayout(self.GridLayout)
    def showmatplot(self):
        dig2 = ShowMatplotlib()
        dig2.exec_()
    def datasplit(self):
        self.browser.clear()
        self.b4.setEnabled(True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)

        X_df = df.drop("target", axis=1)
        y_df = iris_dataset["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_df, y_df, test_size=0.2)

        self.browser.setText("Splitting Iris Dataset was completed!")
        #time.sleep(2)
        self.b5.setEnabled(True)
        self.browser.append("You can Train Dataset from now on")
        self.browser.append("")
        self.browser.append("X_train : " + str(X_train1.shape))
        self.browser.append("y_train : " + str(y_train1.shape))
        self.browser.append("X_test : " + str(X_test1.shape))
        self.browser.append("y_test : " + str(y_test1.shape))
        self.browser.append("")
        self.browser.append("Train Data : " + str(X_train1.shape[0]) + " Instances and "+str(X_train1.shape[1])+" Attributes")
        self.browser.append("Test Data : " + str(X_test1.shape[0]) + " Instances and "+str(X_test1.shape[1])+" Attributes")
    def popup(self):
        ans = QMessageBox.warning(self, "Auto Machine Learning", "It will take a few minutes. Click 'Yes' button to run Auto Machine Learning ",QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            self.autosklearn()
    def autosklearn(self):
        self.browser.clear()
        self.browser.setText("It will take some times. (It takes almost 3 minutes)")
        self.automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120, per_run_time_limit=30
        )

        # Set Data Set
        C = "Categorical"
        N = "Numerical"
        label_name = "target"
        feature_dict = {
            'sepal length (cm)': N,
            'sepal width (cm)': N,
            'petal length (cm)': N,
            'petal width (cm)': N,
        }
        features = df[list(feature_dict.keys())]
        feat_type = list(feature_dict.values())
        labels = df[label_name]
        X_train = self.X_train.values
        X_test = self.X_test.values
        y_train = self.y_train
        y_test = self.y_test
       # X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'])

        self.automl.fit(X_train.copy(), y_train.copy(), feat_type=feat_type, dataset_name='Iris')
        predictions = self.automl.predict(X_test)
        self.browser.append("Auto Sklearn Accuracy score : " + str(sklearn.metrics.accuracy_score(y_test, predictions)*100) + "%")
        self.browser.append("")
        self.browser.append("")
        self.browser.append("More information below...")
        self.browser.append(self.automl.show_models())
        self.browser.append("")
        self.b7.setEnabled(True)
    def pushb7clicked(self):
        dig = saveModel()
        dig.exec_()
        filename = (str(dig.name)+".sav")
        pickle.dump(self.automl, open(filename, "wb"))
        self.browser.append("AutoML Model was saved.")
        self.browser.append("File name : "+str(filename))
    def descrClicked(self, value):
        self.browser.clear()
        self.browser.setText(iris_dataset['DESCR'])
        self.browser.update()
    def moreinfoClicked(self):
        dlg1 = ShowDataFrame()
        dlg1.exec_()
    def TrainModel(self):
        self.browser.clear()
        # Set Data Set
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        X_train1, X_test1, y_train1, y_test1 = X_train.values, X_test.values, y_train, y_test
        self.browser.append("Load Dataset")
        self.browser.append("")
        self.browser.append("")

        # LogisticRegression
        logreg = LogisticRegression()
        logreg.fit(X_train1, y_train1)
        y_pred_logreg = logreg.predict(X_test1)
        acc_log_train = round(logreg.score(X_train1, y_train1) * 100, 2)
        acc_log_test = round(logreg.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Logistic Regression Model>")
        self.browser.append("Train acc : " + str(acc_log_train) + "%")
        self.browser.append("Test acc : "+ str(acc_log_test)+ "%")
        self.browser.append("")
        #time.sleep(3)

        # Support Vector Machine's
        svc = SVC()
        svc.fit(X_train1, y_train1)
        y_pred_svc = svc.predict(X_test1)
        acc_svc_train = round(svc.score(X_train1, y_train1) * 100, 2)
        acc_svc_test = round(svc.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Support Vector Machine's>")
        self.browser.append("Train acc : " + str(acc_svc_train) + "%")
        self.browser.append("Test acc : "+ str(acc_svc_test)+ "%")
        self.browser.append("")
        #time.sleep(3)

        # Naive Bayes
        gaussian = GaussianNB()
        gaussian.fit(X_train1, y_train1)
        y_pred_gau = gaussian.predict(X_test1)
        acc_gau_train = round(gaussian.score(X_train1, y_train1) * 100, 2)
        acc_gau_test = round(gaussian.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Naive Bayes>")
        self.browser.append("Train acc : " + str(acc_gau_train) + "%")
        self.browser.append("Test acc : "+ str(acc_gau_test)+ "%")
        self.browser.append("")

        # K-Nearest Neighbours
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train1, y_train1)
        y_pred_knn = knn.predict(X_test1)
        acc_knn_train = round(knn.score(X_train1, y_train1) * 100, 2)
        acc_knn_test = round(knn.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<K-Nearest Neighbours>")
        self.browser.append("Train acc : " + str(acc_knn_train) + "%")
        self.browser.append("Test acc : "+ str(acc_knn_test)+ "%")
        self.browser.append("")

        # Decision Tree's
        dec = DecisionTreeClassifier()
        dec.fit(X_train1, y_train1)
        y_pred_dec = dec.predict(X_test1)
        acc_dec_train = round(dec.score(X_train1, y_train1) * 100, 2)
        acc_dec_test = round(dec.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Decision Tree's>")
        self.browser.append("Train acc : " + str(acc_dec_train) + "%")
        self.browser.append("Test acc : "+ str(acc_dec_test)+ "%")
        self.browser.append("")

        #sgd
        sgd = SGDClassifier(max_iter=10000)
        sgd.fit(X_train1, y_train1)
        y_pred_sgd = sgd.predict(X_test1)
        acc_sgd_train = round(sgd.score(X_train1, y_train1) * 100, 2)
        acc_sgd_test = round(sgd.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Stochastic Gradient Decent Classifier>")
        self.browser.append("Train acc : " + str(acc_sgd_train) + "%")
        self.browser.append("Test acc : "+ str(acc_sgd_test)+ "%")
        self.browser.append("")

        #Linear SVC
        l_svc = LinearSVC()
        l_svc.fit(X_train1, y_train1)
        y_pred_l_svc = l_svc.predict(X_test1)
        acc_l_svc_train = round(l_svc.score(X_train1, y_train1) * 100, 2)
        acc_l_svc_test = round(l_svc.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Linear Support Vector Machines>")
        self.browser.append("Train acc : " + str(acc_l_svc_train) + "%")
        self.browser.append("Test acc : "+ str(acc_l_svc_test)+ "%")
        self.browser.append("")

        #Perceptron
        per = Perceptron(max_iter=1000)
        per.fit(X_train1, y_train1)
        y_pred_per = per.predict(X_test1)
        acc_per_train = round(per.score(X_train1, y_train1) * 100, 2)
        acc_per_test = round(per.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Perceptron>")
        self.browser.append("Train acc : " + str(acc_per_train) + "%")
        self.browser.append("Test acc : "+ str(acc_per_test)+ "%")
        self.browser.append("")

        #Random Forest
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train1, y_train1)
        y_pred_random_forest = random_forest.predict(X_test1)
        acc_random_forest_train = round(random_forest.score(X_train1, y_train1) * 100, 2)
        acc_random_forest_test = round(random_forest.score(X_test1, y_test1) * 100, 2)
        self.browser.append("<Random Forest>")
        self.browser.append("Train acc : " + str(acc_random_forest_train) + "%")
        self.browser.append("Test acc : "+ str(acc_random_forest_test)+ "%")
        self.browser.append("")

        models = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                      'Random Forest', 'Naive Bayes', 'Perceptron',
                      'Stochastic Gradient Decent', 'Linear SVC',
                      'Decision Tree'],
            'Score': [acc_svc_test, acc_knn_test, acc_log_test,
                      acc_random_forest_test, acc_gau_test, acc_per_test,
                      acc_sgd_test, acc_l_svc_test, acc_dec_test]})
        models.sort_values(by='Score', ascending=True)
        models = PandasModelTrainData(models)
        self.tableView=QTableView()
        self.tableView.setSortingEnabled(True)
        self.tableView.setModel(models)
        self.tableView.setGeometry(850,100,320,400)
        self.tableView.setColumnWidth(0,200)
        self.tableView.sortByColumn(1,Qt.DescendingOrder)
        self.tableView.setWindowTitle("Accuracy")
        self.tableView.show()

'''
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        moreinfo = "/Users/uchan/Projects/keras_talk_py3/text_panda.py"
        subprocess.Popen([python_bin, moreinfo])
'''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.setWindowTitle("Classfication - Custom")
    mywindow.show()
    app.exec_()
