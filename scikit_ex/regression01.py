import sys
import pandas as pd
import random as rnd
import seaborn as sns
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import autosklearn.regression
import autosklearn.classification
import mglearn.datasets

import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sklearn.model_selection import train_test_split
from PandasModel import PandasModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

from sklearn import preprocessing
from sklearn import metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
from sklearn.datasets import load_boston

#X, y = mglearn.datasets.load_extended_boston()
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["MEDV"] = boston.target
import math
df["RM_int"] = df["RM"].map(math.floor)

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
        model = PandasModel(df.drop("RM_int", axis=1))
        self.pandasTv.setModel(model)
        self.pandasTv.setColumnWidth(0,200)

    def loadFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv)");
        self.pathLE.setText(fileName)
        df = pd.read_csv(fileName)
        model = PandasModel(df)
        self.pandasTv.setModel(model)
class ShowMatplotlib(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUI()
    def seabornplot(self):
        self.test = sns.pairplot(df[["MEDV", "RM", "AGE", "DIS", "CRIM"]],size=2)
        return self.test.fig

    def setupUI(self):
        self.setGeometry(200, 200, 1000, 800)
        self.setWindowTitle("PyChart Viewer v0.1")
        self.lineEdit = QLineEdit()
        self.pushButton = QPushButton("Show Fig")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        #self.fig = plt.Figure()
        self.fig1 = self.seabornplot()
        self.canvas = FigureCanvas(self.fig1)
        #self.canvas = FigureCanvas(self.fig)
        #self.axex = self.fig.add_subplot(111)

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
            '''
            self.axex.clear()
            self.fig = sns.kdeplot(df["NOX"], df["LSTAT"])
            self.canvas.draw()
            '''
            self.fig1.clear()
            self.fig1.clear()
            self.fig1 = sns.kdeplot(df["NOX"], df["LSTAT"])
            self.fig1.clear()
            self.fig1 = sns.kdeplot(df["NOX"], df["LSTAT"])
            self.canvas.draw()

        elif code == "1":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV",data = df)
            self.fig1.clear()
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV",data = df)
            self.canvas.draw()

        elif code == "2":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",data = df, orient="v")
            self.fig1.clear()
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",data = df, orient="v")
            self.canvas.draw()

        elif code == "3":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.canvas.draw()

        elif code == "4":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = plt.scatter(df["CRIM"], df["MEDV"])
            plt.xlabel("Per capita crime rate by town (CRIM)")
            plt.ylabel("Housing Price")
            plt.title("Relationship between CRIM and Price")
            self.canvas.draw()

        elif code == "5":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = plt.scatter(df["RM"], df["MEDV"])
            plt.xlabel("Average number of rooms per dwelling(RM)")
            plt.ylabel("Housing Price")
            plt.title("Relationship between RM and Price")
            self.canvas.draw()

        elif code == "6":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = plt.scatter(df["PTRATIO"], df["MEDV"])
            plt.xlabel("Pupil-teacher ratio by town(PTRATIO)")
            plt.ylabel("Housing Price")
            plt.title("Relationship between PTRATIO and Price")
            self.canvas.draw()

        elif code == "7":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = plt.scatter(df["ZN"], df["MEDV"])
            plt.xlabel("proportion of residential land zoned for lots over 25,000 sq.ft.(ZN)")
            plt.ylabel("Housing Price")
            plt.title("Relationship between ZN and Price")
            self.canvas.draw()

        elif code == "8":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = plt.scatter(df["INDUS"], df["MEDV"])
            plt.xlabel("proportion of non-retail business acres per town(INDUS)")
            plt.ylabel("Housing Price")
            plt.title("Relationship between INDUS and Price")
            self.canvas.draw()

        elif code == "9":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = plt.scatter(df["NOX"], df["MEDV"])
            plt.xlabel("nitric oxides concentration (parts per 10 million(NOX)")
            plt.ylabel("Housing Price")
            plt.title("Relationship between NOX and Price")
            self.canvas.draw()

        elif code == "10":
            self.fig1 = sns.violinplot(x="RM_int", y = "MEDV", hue = "CHAS",split = True, data = df)
            self.fig1.clear()
            self.fig1 = sns.regplot(y = "MEDV", x = "RM", data=df, fit_reg=True)
            self.canvas.draw()

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

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
    def setupUI(self):
        self.setGeometry(50, 80, 800, 600)
        #self.browser = QTextEdit()
        self.browser = QTextBrowser()
        self.b1 = QPushButton("Description")
        self.b2 = QPushButton("Check Data Properties")
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
        self.GridLayout.addWidget(self.b6,2,0)
        self.GridLayout.addWidget(self.b3,3,0)
        self.GridLayout.addWidget(self.b4,4,0)
        self.GridLayout.addWidget(self.b5,5,0)
        self.GridLayout.addWidget(self.b7,6,0)
        self.GridLayout.addWidget(self.b8,7,0)
        self.GridLayout.addWidget(self.browser,0,1,8,1)

        self.b4.setEnabled(False)
        self.b5.setEnabled(False)
        self.b7.setEnabled(False)


        self.b1.clicked.connect(self.descrClicked)
        self.b2.clicked.connect(self.moreinfoClicked)
        self.b6.clicked.connect(self.showmatplot)
        self.b3.clicked.connect(self.datasplit)
        self.b4.clicked.connect(self.TrainModel)
        self.b5.clicked.connect(self.popup)
        self.b7.clicked.connect(self.pushb7clicked)
        self.b8.clicked.connect(QCoreApplication.instance().quit)


        self.setLayout(self.GridLayout)
    def showmatplot(self):
        dig2 = ShowMatplotlib()
        dig2.exec_()
    def datasplit(self):
        self.browser.clear()
        self.b4.setEnabled(True)
        df1 = df.drop("RM_int", axis=1)
        X = df1.iloc[:, :-1].values
        y = df1.iloc[:, -1].values
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)
        self.browser.setText("Splitting Iris Dataset was completed!")
        #time.sleep(2)
        self.b5.setEnabled(True)
        X1_df = df.drop("MEDV", axis=1)
        X_df = X1_df.drop("RM_int", axis=1)
        y_df = df["MEDV"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_df, y_df, test_size=0.2)
        #이거 dataframe값이라서 오토에서만 쓸것 -> 근데 다른것에서도 돌아가더랑
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
    def pushb7clicked(self):
        dig = saveModel()
        dig.exec_()
        filename = (str(dig.name)+".sav")
        pickle.dump(self.automl, open(filename, "wb"))
        self.browser.append("AutoML Model was saved.")
        self.browser.append("File name : "+str(filename))
    def descrClicked(self, value):
        self.browser.clear()
        self.browser.setText(boston['DESCR'])
        self.browser.update()
    def moreinfoClicked(self):
        dlg1 = ShowDataFrame()
        dlg1.exec_()
    def TrainModel(self):
        self.browser.clear()
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        X_train1, X_test1, y_train1, y_test1 = X_train.values, X_test.values, y_train.values, y_test.values

        y_train2 = y_train1.reshape(-1, 1)
        y_test2 = y_test1.reshape(-1, 1)

        scalerX = preprocessing.StandardScaler().fit(X_train1)
        scalery = preprocessing.StandardScaler().fit(y_train2)

        X_train3 = scalerX.transform(X_train1)
        X_test3 = scalerX.transform(X_test1)
        y_train3 = scalery.transform(y_train2)
        y_test3 = scalery.transform(y_test2)


        self.browser.append("Load Dataset")
        self.browser.append("")
        self.browser.append("")

        # LinearRegression Model
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        y_pred_lm = lm.predict(X_test)
        acc_lm_train = round(lm.score(X_train, y_train) * 100, 2)
        acc_lm_test = round(lm.score(X_test, y_test) * 100, 2)
        self.browser.append("<LinearRegression Model>")
        self.browser.append("Train acc : " + str(acc_lm_train) + "%")
        self.browser.append("Test acc : "+ str(acc_lm_test)+ "%")
        self.browser.append("")
        #time.sleep(3)

        # Ridge Regression Model
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        acc_ridge_train = round(ridge.score(X_train, y_train) * 100, 2)
        acc_ridge_test = round(ridge.score(X_test, y_test) * 100, 2)
        self.browser.append("<Ridge Regression Model>")
        self.browser.append("Train acc : " + str(acc_ridge_train) + "%")
        self.browser.append("Test acc : "+ str(acc_ridge_test)+ "%")
        self.browser.append("Used Coefficient : "+str(np.sum(ridge.coef_ != 0)))
        self.browser.append("")
        #time.sleep(3)

        # Lasso Regression Model
        lasso = Lasso(alpha=0.1, max_iter=100000)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        acc_lasso_train = round(lasso.score(X_train, y_train) * 100, 2)
        acc_lasso_test = round(lasso.score(X_test, y_test) * 100, 2)
        self.browser.append("<Lasso Regression Model>")
        self.browser.append("Train acc : " + str(acc_lasso_train) + "%")
        self.browser.append("Test acc : "+ str(acc_lasso_test)+ "%")
        self.browser.append("Used Coefficient : "+str(np.sum(lasso.coef_ != 0)))
        self.browser.append("")

        # SGD Regression
        sgd = SGDRegressor(loss="squared_loss", penalty=None, random_state=42, max_iter=100000)
        sgd.fit(X_train3, y_train3)
        y_pred_sgd = sgd.predict(X_test3)
        acc_sgd_train = round(sgd.score(X_train3, y_train3) * 100, 2)
        acc_sgd_test = round(sgd.score(X_test3, y_test3) * 100, 2)
        self.browser.append("<Stochastic Gradient Descent Regression>")
        self.browser.append("Train acc : " + str(acc_sgd_train) + "%")
        self.browser.append("Test acc : "+ str(acc_sgd_test)+ "%")
        self.browser.append("")

        # Decision Tree's
        etr = ExtraTreesRegressor()
        etr.fit(X_train, y_train)
        y_pred_etr = etr.predict(X_test)
        acc_etr_train = round(etr.score(X_train, y_train) * 100, 2)
        acc_etr_test = round(etr.score(X_test, y_test) * 100, 2)
        self.browser.append("<Extra Trees Regressor(Random Forest)>")
        self.browser.append("Train acc : " + str(acc_etr_train) + "%")
        self.browser.append("Test acc : "+ str(acc_etr_test)+ "%")
        self.browser.append("")

        #SVR
        svr = SVR()
        svr.fit(X_train3, y_train3)
        y_pred_svr = svr.predict(X_test3)
        acc_svr_train = round(svr.score(X_train3, y_train3) * 100, 2)
        acc_svr_test = round(svr.score(X_test3, y_test3) * 100, 2)
        self.browser.append("<Support Vector Machine>")
        self.browser.append("Train acc : " + str(acc_svr_train) + "%")
        self.browser.append("Test acc : "+ str(acc_svr_test)+ "%")
        self.browser.append("")


        models = pd.DataFrame({
            'Model': ['LinearRegression', 'Ridge Regression', 'Lasso Regression',
                      'SGD Regression', 'Extra Trees Regressor', 'Support Vector Machine'],
            'Score': [acc_lm_test, acc_ridge_test, acc_lasso_test, acc_sgd_test, acc_etr_test, acc_svr_test]})

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
    def autosklearn(self):
        self.browser.clear()
        self.browser.setText("It will take some times. (It takes almost 3 minutes)")
        self.automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=60, per_run_time_limit=360
        )

        # Set Data Set
        C = 'Categorical'
        N = 'Numerical'
        label_name = 'MEDV'
        feature_dict = {
            'CRIM': N,
            'ZN': N,
            'INDUS': N,
            'CHAS': N,
            'NOX': N,
            'RM': N,
            'AGE': N,
            'DIS': N,
            'RAD': N,
            'TAX': N,
            'PTRATIO': N,
            'B': N,
            'LSTAT': N
        }
        features = df[list(feature_dict.keys())]
        feat_type = list(feature_dict.values())
        feature_types = feat_type
        labels = df[label_name]


        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        self.automl.fit(X_train.copy(), y_train.copy(), feat_type=feat_type, dataset_name='Boston')
        predictions = self.automl.predict(X_test)
        self.browser.append("Auto Sklearn Accuracy score : " + str(self.automl.score(X_test, y_test)) + "%")
        self.browser.append("")
        self.browser.append("")
        self.browser.append("More information below...")
        self.browser.append(self.automl.show_models())
        self.browser.append("")
        self.b7.setEnabled(True)
'''
        python_bin = "/Users/uchan/Projects/keras_talk_py3/myvenv/bin/python"
        moreinfo = "/Users/uchan/Projects/keras_talk_py3/text_panda.py"
        subprocess.Popen([python_bin, moreinfo])
'''
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.setWindowTitle("Regression - House Price in Boston")
    mywindow.show()
    app.exec_()

