# coding=utf-8
"""
    多摄像头版本的主窗口
    @project: EGGRECORDQT
    @Author：lzy
    @file： MainWindow.py
    @date：2024/5/15 9:52
"""
from PyQt5.QtWidgets import QMainWindow, QStackedLayout, QWidget, QMessageBox, QStackedWidget
from views.Label.mainLabel import MainLabel
from views.Label.configLabel import ConfigLabel
from views.Label.multiCameraDetectLabel import MultiCameraDetectLabel
from PyQt5.QtCore import pyqtSlot
from model.utils.getUSB import get_usb_drive_paths
from views.Label.navigationLabel import NavigationLabel


class MainWindow(QMainWindow):
    def __init__(self, app, cfg):
        super().__init__()
        self.app = app
        self.cfg = cfg

        # 使用 QStackedWidget 管理多个界面
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 多摄像头检测界面
        self.detectLabel = MultiCameraDetectLabel(self, cfg)
        self.detectLabel.exitDetectSignal.connect(self.show_navigation)
        self.detectLabel.startNavigationSignal.connect(self.start_navigation_from_detection)
        self.detectLabel.returnChargingSignal.connect(self.return_to_charge_from_detection)
        self.stack.addWidget(self.detectLabel)

        # 导航界面
        self.navigationLabel = NavigationLabel(self, cfg)
        self.navigationLabel.backToDetectSignal.connect(self.show_detect)
        self.navigationLabel.patrolCompletedSignal.connect(self.stop_detection_after_patrol)
        self.stack.addWidget(self.navigationLabel)

        # 默认显示检测界面
        self.stack.setCurrentWidget(self.detectLabel)

        self.initUI()

    def initUI(self):
        """初始化UI"""
        self.setWindowTitle("笼养种鸭产蛋记录系统")
        # 适配 1280x800 显示器布局
        self.resize(1280, 800)
        
        # 设置窗口背景色
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F7F7FA;
            }
        """)
        
        # 运行后自动最大化，保留标题栏和窗口控制按钮
        self.showMaximized()

    @pyqtSlot()
    def show_navigation(self):
        if hasattr(self, 'stack') and hasattr(self, 'navigationLabel'):
            self.stack.setCurrentWidget(self.navigationLabel)

    @pyqtSlot()
    def return_to_charge_from_detection(self):
        if hasattr(self, 'navigationLabel'):
            self.navigationLabel.on_single_navigate("initPoint", "返回充电点")

    @pyqtSlot()
    def start_navigation_from_detection(self):
        if hasattr(self, 'stack') and hasattr(self, 'navigationLabel'):
            if self.navigationLabel.patrol_thread and self.navigationLabel.patrol_thread.isRunning():
                return
            self.stack.setCurrentWidget(self.navigationLabel)
            self.navigationLabel.on_start_patrol()

    @pyqtSlot()
    def stop_detection_after_patrol(self):
        if hasattr(self, 'detectLabel') and getattr(self.detectLabel, 'detection_started', False):
            self.detectLabel._skip_return_after_stop = True
            self.detectLabel.stop_video()

    @pyqtSlot()
    def show_detect(self):
        if hasattr(self, 'stack') and hasattr(self, 'detectLabel'):
            self.stack.setCurrentWidget(self.detectLabel)

    def closeEvent(self, event):
        """主窗口关闭事件处理"""
        # 停止检测界面线程
        if hasattr(self, 'detectLabel'):
            self.detectLabel.exit_detect()
        
        # 停止导航界面线程
        if hasattr(self, 'navigationLabel'):
            self.navigationLabel.close_threads()
            
        event.accept()
