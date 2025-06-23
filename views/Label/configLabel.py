# coding=utf-8
"""
    @project: EGGRECORDQT
    @Author：lzy
    @file： configLabel.py
    @date：2024/5/15 11:29
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFormLayout, QLineEdit, QButtonGroup, QGroupBox, \
    QHBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal
from ..utils.getUSB import get_mounted_usb_paths


def setButtonStyle(mode, button_group):
    # 设置按钮样式
    for btn in button_group.buttons():
        if button_group.id(btn) == mode:
            btn.setStyleSheet("background-color: #ADD8E6;")
        else:
            btn.setStyleSheet("")


class ConfigLabel(QWidget):
    finishConfigSignal = pyqtSignal()

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.parent = parent
        self.cfg = cfg

        usb_paths = get_mounted_usb_paths()
        self.cfg['picture_save_path'] = None
        if len(usb_paths) != 0:
            self.cfg['picture_save_path'] = usb_paths[0]

        self.video_line_edit = QLineEdit(str(self.cfg['video']), self)
        self.match_range_line_edit = QLineEdit(str(self.cfg['match_range']), self)
        self.match_center_line_edit = QLineEdit(str(self.cfg['match_center']), self)
        self.iou_line_edit = QLineEdit(str(self.cfg['iou']), self)
        self.picture_save_path = QLineEdit(str(self.cfg['picture_save_path']), self)
        self.conf_line_edit = QLineEdit(str(self.cfg['conf']), self)
        self.upload_line_edit = QLineEdit(self.cfg['uploadUrl'], self)

        self.initUI()

    def initUI(self):
        # 创建一个垂直布局
        layout = QVBoxLayout()

        # 创建一个表单布局
        form_layout = QFormLayout()

        ############################ 模式选择 ############################
        # 创建一个组容器
        group_box = QGroupBox("模式选择")
        # 在组容器中创建一个垂直布局
        group_layout = QVBoxLayout()
        button_group = QButtonGroup(self)
        # 创建按钮并添加到布局中
        work_mode_button = QPushButton('工作模式', self)
        collection_mode_button = QPushButton('采集模式', self)
        test_model_button = QPushButton('示范模式', self)
        group_layout.addWidget(work_mode_button)
        group_layout.addWidget(collection_mode_button)
        group_layout.addWidget(test_model_button)
        # 将布局设置给组容器
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        button_group.addButton(work_mode_button, id=0)
        button_group.addButton(collection_mode_button, id=1)
        button_group.addButton(test_model_button, id=2)
        # 将按钮组的按钮设置为互斥
        button_group.setExclusive(True)
        # 设置按钮的样式
        setButtonStyle(self.cfg['mode'], button_group)
        # 将按钮组的按钮点击事件连接到槽函数
        button_group.buttonClicked[int].connect(self.onButtonClicked)

        ############################ 上传设置 ############################
        # 创建一个新的组容器用于放置 "上传地址" 和 "自动上传" 按钮
        upload_group_box = QGroupBox("上传设置")
        upload_layout = QVBoxLayout()
        # 创建上传地址行
        upload_label_layout = QHBoxLayout()
        upload_label = QLabel("上传地址:")
        upload_label_layout.addWidget(upload_label)
        upload_label_layout.addWidget(self.upload_line_edit)

        save_label_layout = QHBoxLayout()
        save_label = QLabel("上传地址:")
        save_label_layout.addWidget(save_label)
        save_label_layout.addWidget(self.picture_save_path)

        upload_layout.addLayout(upload_label_layout)
        upload_layout.addLayout(save_label_layout)

        # 创建按钮并添加到表单布局中
        button4 = QPushButton('自动上传', self)
        if self.cfg['upload']:
            button4.setStyleSheet("background-color: #ADD8E6;")
        button4.clicked.connect(self.onButtonUploadClicked)
        upload_layout.addWidget(button4)

        # 将上传设置布局设置给组容器
        upload_group_box.setLayout(upload_layout)
        layout.addWidget(upload_group_box)

        ############################ 目标检测 ############################
        detect_group_box = QGroupBox("目标检测配置")
        detect_layout = QHBoxLayout()

        conf_layout = QHBoxLayout()
        conf_label = QLabel("conf:")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_line_edit)

        iou_layout = QHBoxLayout()
        iou_label = QLabel("IOU:")
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_line_edit)

        detect_layout.addLayout(conf_layout)
        detect_layout.addLayout(iou_layout)

        detect_group_box.setLayout(detect_layout)
        layout.addWidget(detect_group_box)

        ############################ 匹配计数配置 ############################
        match_group_box = QGroupBox("匹配计数配置")
        match_layout = QHBoxLayout()

        match_center_layout = QHBoxLayout()
        match_center_label = QLabel("match_center:")
        match_center_layout.addWidget(match_center_label)
        match_center_layout.addWidget(self.match_center_line_edit)

        match_range_layout = QHBoxLayout()
        match_range_label = QLabel("match_range:")
        match_range_layout.addWidget(match_range_label)
        match_range_layout.addWidget(self.match_range_line_edit)

        match_layout.addLayout(match_center_layout)
        match_layout.addLayout(match_range_layout)

        match_group_box.setLayout(match_layout)
        layout.addWidget(match_group_box)

        ############################ 视频配置 ############################
        video_group_box = QGroupBox("视频采集配置")
        video_layout = QHBoxLayout()

        video_label = QLabel("video:")
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.video_line_edit)

        video_group_box.setLayout(video_layout)
        layout.addWidget(video_group_box)

        ############################ 完成配置 ############################
        # 创建完成按钮
        finish_button = QPushButton("完成", self)
        finish_button.clicked.connect(self.finishConfig)
        form_layout.addRow(finish_button)

        # 将表单布局添加到主布局中
        layout.addLayout(form_layout)

        # 设置布局到窗口部件
        self.setLayout(layout)

    def onButtonClicked(self, id):
        # 按钮点击事件处理函数
        # 更新模式值，并设置按钮样式
        self.cfg['mode'] = id  # 引用传递，修改外部也同步修改
        button = self.sender()
        setButtonStyle(id, button)

    def finishConfig(self):
        self.cfg['video'] = int(self.video_line_edit.text())
        self.cfg['match_range'] = int(self.match_range_line_edit.text())
        self.cfg['match_center'] = int(self.match_center_line_edit.text())
        self.cfg['IOU'] = float(self.iou_line_edit.text())
        self.cfg['conf'] = float(self.conf_line_edit.text())
        self.cfg['uploadUrl'] = self.upload_line_edit.text()
        self.cfg['picture_save_path'] = self.picture_save_path.text()
        self.finishConfigSignal.emit()

    def onButtonUploadClicked(self):
        self.cfg['upload'] = not self.cfg['upload']
        button = self.sender()
        if self.cfg['upload']:
            button.setStyleSheet("background-color: #ADD8E6;")
        else:
            button.setStyleSheet("")  # 清除按钮样式
