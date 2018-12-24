import numpy as np
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, QTimer, QSize
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import mainFile


def center_the_window(w):
    screen = QDesktopWidget().screenGeometry()
    widget = w.geometry()
    x = screen.width() / 2
    y = screen.height() / 2
    # x = screen.width() - widget.width()
    # y = screen.height() - widget.height()

    widget.height(500)
    widget.width(500)
    w.move(x, y)


class MainWindow():
    def __init__(self):
        self.title = 'Welcome to Cartoonization!'
        # '/media/kamal/New Volume/CUFE/Image Processing/Project/cartoon.jpg'
        self.logoPath = '/media/aosman/New Volume/image processing/finalproject/logo.jpg'
        self.tooltipText = "Developed by Kamal, Belal, Osman"
        self.startButtonText = "Start"
        self.backButtonText = "Back"

        self.mainWindow = QWidget()
        self.mainWindow.setObjectName('mainWindow')
        self.mainLayout = QVBoxLayout()
        self.mainLayout.setContentsMargins(0,0,0,0)


        self.labelLogo = QLabel()
        self.pixmap = QPixmap(self.logoPath)
        self.labelLogo.setPixmap(self.pixmap)
        self.labelLogo.setToolTip(self.tooltipText)
        self.labelLogo.setObjectName('mainLogo')

        self.labelLogo.setScaledContents(1)
        self.labelLogo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.mainLayout.addWidget(self.labelLogo,13)

        startButton = QPushButton(self.startButtonText)
        startButton.setCursor(Qt.PointingHandCursor)
        startButton.clicked.connect(self.startApp)
        startButton.setObjectName('mainStartButton')
        startButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


        self.mainLayout.addWidget(startButton,2)



        self.mainWindow.setLayout(self.mainLayout)

        # MIN SIZE TO BE CHANGED
        # self.mainWindow.setMinimumSize(QSize(1280,720))

        self.mainWindow.setWindowTitle(self.title)
        self.mainWindow.setFixedSize(QSize(1280,720))
        # center_the_window(self.mainWindow)
        self.mainWindow.show()


    def startApp(self):
        self.appWindow = QWidget()
        self.appLayout = QVBoxLayout()


        backButton = QPushButton(self.backButtonText)
        backButton.setCursor(Qt.PointingHandCursor)
        backButton.setObjectName('backButton')
        backButton.clicked.connect(self.backToMain)

        self.appLayout.addWidget(backButton)

        horizontalLayout1 = QHBoxLayout()
        uploadLabel = QLabel("Upload Image")
        uploadLabel.setObjectName('uploadLabel')
        self.uploadButton = QPushButton("Browse")
        self.uploadButton.setObjectName('uploadButton')
        self.uploadButton.clicked.connect(self.browseImage)
        self.uploadButton.setCursor(Qt.PointingHandCursor)
        horizontalLayout1.addWidget(uploadLabel)
        horizontalLayout1.addWidget(self.uploadButton)


        horizontalLayout2 = QHBoxLayout()
        dropdownLabel = QLabel("Select Mode")
        dropdownLabel.setObjectName('dropdownLabel')
        self.dropdownCombobox = QComboBox()
        self.dropdownCombobox.setCursor(Qt.PointingHandCursor)
        self.dropdownCombobox.setObjectName('dropdownCombobox')
        self.dropdownCombobox.addItem("Cartoonization")


        horizontalLayout2.addWidget(dropdownLabel)
        horizontalLayout2.addWidget(self.dropdownCombobox)


        self.appLayout.addLayout(horizontalLayout1)
        self.appLayout.addLayout(horizontalLayout2)


        startButton = QPushButton(self.startButtonText)
        startButton.setCursor(Qt.PointingHandCursor)
        startButton.setObjectName('appStartButton')
        # startButton.resize(10,10)
        startButton.clicked.connect(self.startFunc)
        self.appLayout.addWidget(startButton)


        self.appWindow.setLayout(self.appLayout)

        # MIN SIZE TO BE CHANGED
        # self.window.setMinimumSize(QSize(1920,1080))

        self.appWindow.setWindowTitle(self.title)

        self.mainWindow.hide()
        # center_the_window(self.appWindow)
        self.appWindow.setFixedSize(QSize(1280, 720))
        self.appWindow.show()


    def backToMain(self):
        self.appWindow.hide()
        # center_the_window(self.mainWindow)
        self.mainWindow.show()



    def browseImage(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.appWindow, 'Select image', QtCore.QDir.rootPath(), '*')
        if hasattr(self, 'fileName') == True and self.fileName != "":
            self.uploadButton.setText(self.fileName.split('/')[-1])
        # fileName is the path of the file

    def startFunc(self):
        self.mode = str(self.dropdownCombobox.currentText())
        if hasattr(self, 'fileName') == False or self.fileName == "":
            messageBox = QMessageBox.about(self.appWindow, "Error!", "Please select an image first!")

        else:
            if self.mode == "Cartoonization":
                ######################################## CALL MAIN LOGIC FUNCTION HERE ###########################################################
                mainFile.mainfunc(self.fileName)

app = QApplication([])
# app.setOverrideCursor(Qt.PointingHandCursor)
app.setStyle('Fusion')

# Changing color palette of current theme
palette = QPalette()
palette.setColor(QPalette.ButtonText, Qt.white)
palette.setColor(QPalette.Button, Qt.transparent)
palette.setColor(QPalette.Background, QColor('#fffdf6'))

app.setPalette(palette)

# Custom CSS
# #8b104e
app.setStyleSheet("QPushButton { border: 1px solid #fffdf6; border-radius: 30px; background-color:#96275f; } QComboBox { border: 1px solid #fffdf6; border-radius: 30px; background-color: #96275f } QMessageBox { background-color:#fffdf6 }   QLabel#uploadLabel { margin: 0 0 0 200px; font-size: 30px; color: #8b104e } QLabel#dropdownLabel { margin: 0 0 0 200px; font-size: 30px; color: #8b104e } QPushButton:hover { background-color:#8b104e;}")


mainWindow = MainWindow()

exec_ = app.exec_()
