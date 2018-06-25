from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
import interact
import time
from termcolor import colored
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(5, 80, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(3, 0, 591, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(5, 30, 561, 41))
        self.label_4.setObjectName("label_4")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(5, 100, 631, 141))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(5, 270, 631, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(5, 250, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(5, 350, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(5, 370, 631, 51))
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 320, 80, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.on_click)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 320, 80, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.on_click2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 20))
        self.menubar.setObjectName("menubar")
        self.menuDemo_by_Jin_Man_Park = QtWidgets.QMenu(self.menubar)
        self.menuDemo_by_Jin_Man_Park.setObjectName("menuDemo_by_Jin_Man_Park")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuDemo_by_Jin_Man_Park.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Text-Fusion LSTM Interactive Session"))
        self.label_2.setText(_translate("MainWindow", "Document"))
        self.label_3.setText(_translate("MainWindow", "Text-Fusion LSTM for Question Answering Demo"))
        self.label_4.setText(_translate("MainWindow", "Insert a paragraph and write your own question.\n"
"The answer is extracted from the given paragraph, by the Text-Fusion LSTM."))
        self.label_5.setText(_translate("MainWindow", "Question"))
        self.label_6.setText(_translate("MainWindow", "Answer"))
        self.pushButton.setText(_translate("MainWindow", "submit"))
        self.pushButton_2.setText(_translate("MainWindow", "clear"))
        self.menuDemo_by_Jin_Man_Park.setTitle(_translate("MainWindow", "demo by Jin-Man Park"))

    #@pyqtSlot()
    def on_click(self):
        evidence = self.textEdit.toPlainText()
        print(evidence.strip())
        question = self.textEdit_2.toPlainText()
        print(question.strip())
        try:
            rescode_e, evidence, lang_e = interact.translate(evidence.strip(), 'en')
            rescode_q, question, lang_q = interact.translate(question.strip(), 'en')
        except:
            lang_q = 'en'
            lang_e = 'en'
            pass
        start_time = time.time()
        try:
            prediction=interact.infer.response(evidence,question)
            print(lang_q)
            rescode_p, prediction, _ = interact.translate(prediction,lang_q,verbose=False)
            print(prediction)
        except Exception as e:
            print(e)
            print('Error  : ', 'red', 'Please try again with another natural language input. \n')
            pass
        end_time = time.time()
        print(colored('Time    : ','green'),'{:.4f}s'.format(end_time - start_time))
        print(colored('Answer  : ','green'),'{}'.format(prediction),'\n')
        self.textEdit_3.setPlainText(prediction)

    def on_click2(self):
        self.textEdit.setPlainText("")
        self.textEdit_2.setPlainText("")
        self.textEdit_3.setPlainText("")
