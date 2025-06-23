# coding=utf-8
"""
    @project: EGGRECORDQT
    @Author：wjt
    @file： mainLabel.py
    @date：2024/5/15 10:17
"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QIcon
from ..utils.getUSB import get_mounted_usb_paths
from ..utils.getUSB import get_usb_drive_paths


class MainLabel(QWidget):
    showConfigSignal = pyqtSignal()
    showDetectSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.background_image = QPixmap("resources/back.png")
        self.initUI()

    def initUI(self):
        # 创建一个垂直布局
        layout = QHBoxLayout()

        # 创建按钮和标签
        button1 = QPushButton(self)
        button2 = QPushButton(self)
        button3 = QPushButton(self)

        button3.clicked.connect(lambda: self.showConfigSignal.emit())
        button1.clicked.connect(lambda: self.showDetectSignal.emit())
        button2.clicked.connect(self.showUSB)

        # 创建图标和文本的垂直布局
        self.addIconAndText(button1, QIcon("resources/flight.png"), "开始巡检")
        self.addIconAndText(button2, QIcon("resources/inspection.png"), "环境检查")
        self.addIconAndText(button3, QIcon("resources/config.png"), "设置")

        # 将按钮添加到布局中
        layout.addWidget(button2)
        layout.addWidget(button1)
        layout.addWidget(button3)

        # 设置布局到窗口部件
        self.setLayout(layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background_image)

    def addIconAndText(self, button, icon, text):
        layout = QVBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(120, 120))
        icon_label.setAlignment(Qt.AlignCenter)
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label, alignment=Qt.AlignCenter)
        layout.addWidget(text_label, alignment=Qt.AlignCenter)
        button.setLayout(layout)
        button.setStyleSheet("background-color: skyblue; border-radius: 20px;")
        button.setFixedSize(200, 200)

    def showUSB(self):
        #usb_paths = get_mounted_usb_paths()     #linux
        usb_paths = get_usb_drive_paths()
        if len(usb_paths) == 0:
            QMessageBox.information(self, "提示", '检测不到U盘')
        else:
            QMessageBox.information(self, "提示", str(usb_paths))
