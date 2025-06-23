# coding=utf-8
"""
    多摄像头版本的主窗口
    @project: EGGRECORDQT
    @Author：lzy
    @file： MainWindow.py
    @date：2024/5/15 9:52
"""
from PyQt5.QtWidgets import QMainWindow, QStackedLayout, QWidget, QMessageBox
from views.Label.mainLabel import MainLabel
from views.Label.configLabel import ConfigLabel
from views.Label.multiCameraDetectLabel import MultiCameraDetectLabel
from PyQt5.QtCore import pyqtSlot
from model.utils.getUSB import get_usb_drive_paths


class MainWindow(QMainWindow):
    def __init__(self, app, cfg):
        super().__init__()
        self.app = app
        self.cfg = cfg

        # 创建主窗口的中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主窗口的布局
        self.stackedLayout = QStackedLayout(central_widget)

        # 创建三个主要界面
        self.mainLabel = MainLabel(self)
        self.configLabel = ConfigLabel(self, cfg)
        self.detectLabel = MultiCameraDetectLabel(self, cfg)  # 使用多摄像头检测界面

        # 连接信号
        self.mainLabel.showConfigSignal.connect(self.showConfigPage)
        self.mainLabel.showDetectSignal.connect(self.showDetectPage)
        self.configLabel.finishConfigSignal.connect(self.finishConfigPage)
        self.detectLabel.exitDetectSignal.connect(self.exitDetectPage)

        # 添加界面到堆叠布局
        self.stackedLayout.addWidget(self.mainLabel)
        self.stackedLayout.addWidget(self.configLabel)
        self.stackedLayout.addWidget(self.detectLabel)

        self.initUI()

    def initUI(self):
        """初始化UI"""
        self.setWindowTitle("多摄像头笼养蛋鸭产蛋记录系统")
        self.setFixedSize(1200, 800)  # 增大窗口尺寸以容纳多摄像头显示
        self.showMainPage()

    def showMainPage(self):
        """显示主页"""
        self.stackedLayout.setCurrentIndex(0)

    @pyqtSlot()
    def showConfigPage(self):
        """显示配置页面"""
        self.stackedLayout.setCurrentIndex(1)

    @pyqtSlot()
    def exitDetectPage(self):
        """退出检测页面，返回主页"""
        self.stackedLayout.setCurrentIndex(0)

    @pyqtSlot()
    def showDetectPage(self):
        """显示检测页面"""
        # 检查采集模式下是否有U盘连接
        if self.cfg['mode'] == 1:
            usb_paths = get_usb_drive_paths()
            if not usb_paths:
                QMessageBox.information(self, "提示", '检测不到U盘，请先插入U盘')
                return
            # 更新U盘路径
            self.cfg['picture_save_path'] = usb_paths[0]

        # 显示检测页面
        self.stackedLayout.setCurrentIndex(2)

    @pyqtSlot()
    def finishConfigPage(self):
        """完成配置，返回主页"""
        QMessageBox.information(self, "提示", "配置已保存！")
        self.stackedLayout.setCurrentIndex(0)
