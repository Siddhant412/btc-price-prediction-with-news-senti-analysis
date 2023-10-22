# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mark-2.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, price, email_main, pwd_main, senti_val):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(716, 398)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 210, 131, 41))
        self.pushButton_2.setObjectName("pushButton_2")

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(270, 90, 411, 41))
        self.lineEdit.setObjectName("lineEdit")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 100, 171, 16))
        font = QtGui.QFont()
        font.setPointSize(12)

        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(195, 20, 371, 51))
        font = QtGui.QFont()
        font.setPointSize(16)

        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(190, 220, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(12)

        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(190, 280, 181, 20))
        font = QtGui.QFont()
        font.setPointSize(12)

        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(380, 270, 131, 41))
        self.pushButton_3.setObjectName("pushButton_3")

        flag = 1

        while(flag):

            email, done2 = QtWidgets.QInputDialog.getText(self.MainWindow, 'Input Dialog', 'Enter your Email ID: ')
            #print(email)

            if email =="":
                sys.exit()

            pwd, done3 = QtWidgets.QInputDialog.getText(self.MainWindow, 'Input Dialog', 'Enter your Password: ')
            #print(pwd)

            if email == email_main and pwd == pwd_main:
                flag = 0
            
            if pwd == "":
                sys.exit()

        news, done1 = QtWidgets.QInputDialog.getText(self.MainWindow, 'Input Dialog', 'Enter the news here: ')
        #print(news)

        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow, price, news, senti_val)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        return news

    def retranslateUi(self, MainWindow, price, news, senti_val):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", price))
        self.label.setText(_translate("MainWindow", "Enter news here"))
        self.label_2.setText(_translate("MainWindow", "Bitcoin Price Prediction System"))
        self.label_3.setText(_translate("MainWindow", "Predicted Price"))
        self.label_4.setText(_translate("MainWindow", "Sentiment of news"))
        self.pushButton_3.setText(_translate("MainWindow", senti_val))
        self.lineEdit.setText(_translate("MainWindow", news))


def run_gui(price, email, pwd, senti_val):
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    news = ui.setupUi(MainWindow, price, email, pwd, senti_val)
    MainWindow.show()
    sys.exit(app.exec_())
