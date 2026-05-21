# coding=utf-8
"""
    @project: EGGRECORDQT
    @Author：wjt
    @file： detectLabel.py
    @date：2024/5/15 11:26
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QGroupBox, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon
from model.QTInterface import Interface
from model.QTCapture import CaptureInterface
import os
from model.utils.path_utils import resource_path


def _res(name: str) -> str:
    return resource_path("resources", name)


class DetectLabel(QWidget):
    exitDetectSignal = pyqtSignal()

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.button5 = QPushButton(self)
        self.button4 = QPushButton(self)
        self.button_layout = QVBoxLayout()
        self.button3 = QPushButton(self)
        self.button2 = QPushButton(self)
        self.button1 = QPushButton(self)
        self.config = cfg
        self.image_label = QLabel()
        self.frame_producer = None
        self.initUI()

    def initUI(self):
        # 创建一个垂直布局
        layout = QGridLayout()

        group_box = QGroupBox("视频检测")

        hbox_layout = QHBoxLayout()

        # 创建用于显示图片的 QLabel
        self.image_label.setStyleSheet("background-color: gray;")  # 设置背景颜色为灰色
        self.image_label.setText("Placeholder for Image")
        self.image_label.setFixedSize(900, 600)

        self.button1.clicked.connect(self.start_video)
        self.button2.clicked.connect(self.pause_video)
        self.button3.clicked.connect(self.resume_video)
        self.button4.clicked.connect(self.stop_video)
        self.button5.clicked.connect(self.exit_detect)

        self.button_layout.addWidget(self.button1)
        self.button_layout.addWidget(self.button2)
        self.button_layout.addWidget(self.button3)
        self.button_layout.addWidget(self.button4)
        self.button_layout.addWidget(self.button5)

        # 创建图标和文本的垂直布局
        self.addIconAndText(self.button1, QIcon(_res("flight.png")), "开始检测")
        self.addIconAndText(self.button2, QIcon(_res("pause-one.svg")), "暂停检测")
        self.addIconAndText(self.button3, QIcon(_res("undo.svg")), "恢复检测")
        self.addIconAndText(self.button4, QIcon(_res("clear-format.svg")), "停止检测")
        self.addIconAndText(self.button5, QIcon(_res("logout.png")), "退出界面")

        # 将 QLabel 和按钮垂直布局放置在 hbox_layout 中
        hbox_layout.addLayout(self.button_layout)
        hbox_layout.addWidget(self.image_label)

        # 将 hbox_layout 添加到 QGroupBox 内的垂直布局中
        group_layout = QVBoxLayout()
        group_layout.addLayout(hbox_layout)
        group_box.setLayout(group_layout)

        # 将组容器添加到主垂直布局中
        layout.addWidget(group_box)

        # 设置布局到窗口部件
        self.setLayout(layout)

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        # 在QLabel中显示图像
        self.image_label.setPixmap(pixmap)

    def pause_video(self):
        self.set_button_color(self.button2)
        if self.frame_producer is not None:
            self.frame_producer.pause()

    # def start_video(self):   #Linux
    #     self.set_button_color(self.button1)
    #     # 检查是否需要创建新的视频流处理对象
    #     if self.frame_producer is None:
    #         if self.config['mode'] == 1 and os.path.exists(self.config['picture_save_path']) is False:
    #             QMessageBox.information(self, "提示", '检测不到U盘')
    #             return
    #         # 选择视频处理接口类
    #         interface_class = CaptureInterface if self.config['mode'] == 1 else Interface
    #         self.frame_producer = interface_class(self.config)
    #         self.frame_producer.frame_generated.connect(self.update_frame)
    #         self.frame_producer.start()
    def start_video(self):
        self.set_button_color(self.button1)
        # 检查是否需要创建新的视频流处理对象
        if self.frame_producer is None:
            if self.config['mode'] == 1:
                # 使用Windows专用函数
                from ..utils.getUSB import get_usb_drive_paths
                usb_paths = get_usb_drive_paths()
                if not usb_paths:
                    QMessageBox.information(self, "提示", '检测不到U盘')
                    return
                # 确保使用检测到的U盘路径
                self.config['picture_save_path'] = usb_paths[0]

            # 选择视频处理接口类
            interface_class = CaptureInterface if self.config['mode'] == 1 else Interface
            self.frame_producer = interface_class(self.config)
            self.frame_producer.frame_generated.connect(self.update_frame)
            self.frame_producer.start()

    def resume_video(self):
        self.set_button_color(self.button5)
        self.set_button_color(self.button3)
        if self.frame_producer is not None:
            self.frame_producer.resume()

    def exit_detect(self):
        self.set_button_color(self.button5)
        if self.frame_producer is not None:
            self.frame_producer.stop()
            self.frame_producer.wait()
            self.frame_producer = None
        self.exitDetectSignal.emit()

    def stop_video(self):
        self.set_button_color(self.button4)
        if self.frame_producer is not None:
            self.frame_producer.stop()
            self.frame_producer.wait()
            self.frame_producer = None

    def addIconAndText(self, button, icon, text):
        layout = QVBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(50, 50))
        icon_label.setAlignment(Qt.AlignCenter)
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label, alignment=Qt.AlignCenter)
        layout.addWidget(text_label, alignment=Qt.AlignCenter)
        button.setLayout(layout)
        button.setStyleSheet("background-color: skyblue; border-radius: 20px;")
        button.setFixedSize(100, 100)

    def set_button_color(self, active_button):
        for button in [self.button1, self.button2, self.button3, self.button4, self.button5]:
            if button == active_button:
                button.setStyleSheet("background-color: lightgreen; border-radius: 20px;")
            else:
                button.setStyleSheet("background-color: skyblue; border-radius: 20px;")
