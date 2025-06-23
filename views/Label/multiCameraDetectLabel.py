# coding=utf-8
"""
    多摄像头检测界面（改进版）
    @project: EGGRECORDQT
    @Author：lzy
    @file： multiCameraDetectLabel.py
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
                             QLabel, QGroupBox, QGridLayout, QMessageBox,
                             QTextEdit, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QFont
from model.MultiCameraInterface import MultiCameraInterface
import os
import psutil
import time


class MultiCameraDetectLabel(QWidget):
    exitDetectSignal = pyqtSignal()

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.config = cfg
        self.frame_producer = None
        self.camera_count = cfg.get('camera_count', 6)
        self.image_labels = []
        self.status_labels = []  # 添加状态标签
        self.start_time = None

        # 添加系统监控
        self.system_monitor_timer = QTimer()
        self.system_monitor_timer.timeout.connect(self.update_system_status)

        self.initUI()

    def initUI(self):
        # 创建主分割器
        main_splitter = QSplitter(Qt.Vertical)

        # 创建控制面板
        control_widget = self.create_control_panel()
        main_splitter.addWidget(control_widget)

        # 创建视频显示区域
        video_widget = self.create_video_panel()
        main_splitter.addWidget(video_widget)

        # 创建状态面板
        status_widget = self.create_status_panel()
        main_splitter.addWidget(status_widget)

        # 设置分割器比例
        main_splitter.setStretchFactor(0, 1)  # 控制面板
        main_splitter.setStretchFactor(1, 4)  # 视频面板
        main_splitter.setStretchFactor(2, 1)  # 状态面板

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

    def create_control_panel(self):
        """创建控制面板"""
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        # 创建控制按钮
        self.start_button = self.create_control_button("开始检测", "resources/flight.png")
        self.pause_button = self.create_control_button("暂停检测", "resources/pause-one.svg")
        self.resume_button = self.create_control_button("恢复检测", "resources/undo.svg")
        self.stop_button = self.create_control_button("停止检测", "resources/clear-format.svg")
        self.exit_button = self.create_control_button("退出界面", "resources/logout.png")

        # 连接按钮信号
        self.start_button.clicked.connect(self.start_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.resume_button.clicked.connect(self.resume_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.exit_button.clicked.connect(self.exit_detect)

        # 添加按钮到控制布局
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.resume_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.exit_button)

        # 添加系统状态显示
        self.system_status_label = QLabel("系统状态: 就绪")
        self.system_status_label.setStyleSheet("color: green; font-weight: bold;")
        control_layout.addWidget(self.system_status_label)

        control_group.setLayout(control_layout)
        return control_group

    def create_video_panel(self):
        """创建视频显示面板"""
        video_group = QGroupBox("多摄像头监控")
        video_layout = QGridLayout()

        # 计算网格布局的行列数
        if self.camera_count <= 3:
            rows, cols = 1, self.camera_count
        elif self.camera_count <= 6:
            rows, cols = 2, 3
        elif self.camera_count <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = (self.camera_count + 3) // 4, 4

        # 创建图像标签和状态标签
        for i in range(self.camera_count):
            # 创建摄像头容器
            camera_container = QWidget()
            camera_layout = QVBoxLayout()
            camera_layout.setContentsMargins(2, 2, 2, 2)

            # 创建图像标签
            label = QLabel()
            label.setStyleSheet("background-color: gray; border: 2px solid darkgray;")
            label.setAlignment(Qt.AlignCenter)
            label.setText(f"摄像头 {i + 1}")
            label.setMinimumSize(320, 240)
            self.image_labels.append(label)

            # 创建状态标签
            status_label = QLabel(f"摄像头 {i + 1}: 未连接")
            status_label.setStyleSheet("color: red; font-size: 12px; background-color: white; padding: 2px;")
            status_label.setAlignment(Qt.AlignCenter)
            self.status_labels.append(status_label)

            # 添加到容器
            camera_layout.addWidget(label)
            camera_layout.addWidget(status_label)
            camera_container.setLayout(camera_layout)

            # 计算位置
            row = i // cols
            col = i % cols
            video_layout.addWidget(camera_container, row, col)

        video_group.setLayout(video_layout)
        return video_group

    def create_status_panel(self):
        """创建状态面板"""
        status_group = QGroupBox("系统状态")
        status_layout = QHBoxLayout()

        # 运行时间显示
        self.runtime_label = QLabel("运行时间: 00:00:00")
        self.runtime_label.setFont(QFont("Arial", 10, QFont.Bold))

        # CPU使用率
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)
        self.cpu_progress.setFormat("CPU: %p%")

        # 内存使用率
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        self.memory_progress.setFormat("内存: %p%")

        # 日志显示
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(100)
        self.log_display.setPlainText("系统日志:\n")

        status_layout.addWidget(self.runtime_label)
        status_layout.addWidget(self.cpu_progress)
        status_layout.addWidget(self.memory_progress)
        status_layout.addWidget(self.log_display)

        status_group.setLayout(status_layout)
        return status_group

    def create_control_button(self, text, icon_path):
        """创建带有图标和文本的控制按钮"""
        button = QPushButton()
        layout = QVBoxLayout()

        # 加载图标
        icon_label = QLabel()
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            icon_label.setPixmap(icon.pixmap(30, 30))
        else:
            icon_label.setText("●")  # 备用图标
            icon_label.setStyleSheet("font-size: 30px;")
        icon_label.setAlignment(Qt.AlignCenter)

        # 创建文本标签
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setFont(QFont("Arial", 9))

        # 添加到布局
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.setContentsMargins(5, 5, 5, 5)

        # 设置按钮布局
        button.setLayout(layout)
        button.setStyleSheet(
            "QPushButton {"
            "background-color: skyblue; "
            "border-radius: 10px; "
            "min-width: 80px; "
            "min-height: 80px; "
            "}"
        )

        return button

    def update_frames(self, images):
        """更新所有摄像头的画面"""
        for i, image in enumerate(images):
            if i < len(self.image_labels):
                try:
                    pixmap = QPixmap.fromImage(image)
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(
                            self.image_labels[i].width(),
                            self.image_labels[i].height(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.image_labels[i].setPixmap(pixmap)
                        self.image_labels[i].setText("")  # 清除文本

                        # 更新状态标签
                        if i < len(self.status_labels):
                            self.status_labels[i].setText(f"摄像头 {i + 1}: 正常")
                            self.status_labels[i].setStyleSheet(
                                "color: green; font-size: 12px; background-color: white; padding: 2px;")
                    else:
                        # 图像无效，显示连接异常
                        if i < len(self.status_labels):
                            self.status_labels[i].setText(f"摄像头 {i + 1}: 异常")
                            self.status_labels[i].setStyleSheet(
                                "color: red; font-size: 12px; background-color: white; padding: 2px;")
                except Exception as e:
                    print(f"更新摄像头 {i} 画面异常: {e}")
                    if i < len(self.status_labels):
                        self.status_labels[i].setText(f"摄像头 {i + 1}: 错误")
                        self.status_labels[i].setStyleSheet(
                            "color: red; font-size: 12px; background-color: white; padding: 2px;")

    def update_system_status(self):
        """更新系统状态"""
        try:
            # 更新运行时间
            if self.start_time:
                elapsed = int(time.time() - self.start_time)
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                self.runtime_label.setText(f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # 更新CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_progress.setValue(int(cpu_percent))

            # 更新内存使用率
            memory_percent = psutil.virtual_memory().percent
            self.memory_progress.setValue(int(memory_percent))

            # 根据资源使用情况更新系统状态
            if cpu_percent > 85 or memory_percent > 85:
                self.system_status_label.setText("系统状态: 高负载")
                self.system_status_label.setStyleSheet("color: red; font-weight: bold;")
                self.add_log(f"警告: 系统负载过高 (CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%)")
            elif cpu_percent > 70 or memory_percent > 70:
                self.system_status_label.setText("系统状态: 中等负载")
                self.system_status_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.system_status_label.setText("系统状态: 正常")
                self.system_status_label.setStyleSheet("color: green; font-weight: bold;")

        except Exception as e:
            print(f"更新系统状态异常: {e}")

    def add_log(self, message):
        """添加日志信息"""
        current_time = time.strftime("%H:%M:%S")
        log_message = f"[{current_time}] {message}\n"
        self.log_display.append(log_message)

        # 限制日志长度
        if len(self.log_display.toPlainText()) > 2000:
            lines = self.log_display.toPlainText().split('\n')
            self.log_display.setPlainText('\n'.join(lines[-20:]))

    def set_button_color(self, active_button):
        """设置按钮颜色以突出显示活动按钮"""
        for button in [self.start_button, self.pause_button, self.resume_button,
                       self.stop_button, self.exit_button]:
            if button == active_button:
                button.setStyleSheet(
                    "QPushButton {"
                    "background-color: lightgreen; "
                    "border-radius: 10px; "
                    "min-width: 80px; "
                    "min-height: 80px; "
                    "border: 2px solid green;"
                    "}"
                )
            else:
                button.setStyleSheet(
                    "QPushButton {"
                    "background-color: skyblue; "
                    "border-radius: 10px; "
                    "min-width: 80px; "
                    "min-height: 80px; "
                    "}"
                )

    def start_video(self):
        """启动视频处理"""
        try:
            self.set_button_color(self.start_button)
            self.add_log("开始启动视频处理...")

            # 检查是否需要创建新的视频流处理对象
            if self.frame_producer is None:
                # 检查U盘是否连接（仅采集模式需要）
                if self.config['mode'] == 1:
                    from model.utils.getUSB import get_usb_drive_paths
                    usb_paths = get_usb_drive_paths()
                    if not usb_paths:
                        QMessageBox.information(self, "提示", '检测不到U盘')
                        self.add_log("错误: 检测不到U盘")
                        return
                    # 确保使用检测到的U盘路径
                    self.config['picture_save_path'] = usb_paths[0]
                    self.add_log(f"使用U盘路径: {usb_paths[0]}")

                self.add_log("创建多摄像头处理接口...")
                # 创建多摄像头处理接口
                self.frame_producer = MultiCameraInterface(self.config)
                self.frame_producer.frames_generated.connect(self.update_frames)

                self.add_log("启动多摄像头处理...")
                self.frame_producer.start()

                # 记录启动时间
                self.start_time = time.time()

                # 启动系统监控
                self.system_monitor_timer.start(1000)  # 每秒更新

                self.add_log("多摄像头处理已启动")

        except Exception as e:
            import traceback
            error_msg = f"启动失败: {str(e)}"
            print(f"启动视频处理异常: {e}")
            traceback.print_exc()
            self.add_log(f"错误: {error_msg}")
            QMessageBox.critical(self, "错误", error_msg)

    def pause_video(self):
        """暂停视频处理"""
        self.set_button_color(self.pause_button)
        if self.frame_producer is not None:
            self.frame_producer.pause()
            self.add_log("视频处理已暂停")

    def resume_video(self):
        """恢复视频处理"""
        self.set_button_color(self.resume_button)
        if self.frame_producer is not None:
            self.frame_producer.resume()
            self.add_log("视频处理已恢复")

    def stop_video(self):
        """停止视频处理"""
        self.set_button_color(self.stop_button)
        if self.frame_producer is not None:
            self.add_log("正在停止视频处理...")
            self.frame_producer.stop()
            self.frame_producer.wait(5000)  # 5秒超时
            self.frame_producer = None

            # 停止系统监控
            self.system_monitor_timer.stop()
            self.start_time = None

            # 清空所有图像显示
            for i, label in enumerate(self.image_labels):
                label.clear()
                label.setText(f"摄像头 {i + 1}")

            # 重置状态标签
            for i, status_label in enumerate(self.status_labels):
                status_label.setText(f"摄像头 {i + 1}: 未连接")
                status_label.setStyleSheet("color: gray; font-size: 12px; background-color: white; padding: 2px;")

            self.add_log("视频处理已停止")

    def exit_detect(self):
        """退出检测界面"""
        self.set_button_color(self.exit_button)
        if self.frame_producer is not None:
            self.add_log("正在退出检测界面...")
            self.frame_producer.stop()
            self.frame_producer.wait(5000)
            self.frame_producer = None

        # 停止系统监控
        self.system_monitor_timer.stop()

        self.add_log("已退出检测界面")
        self.exitDetectSignal.emit()

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.frame_producer is not None:
            self.frame_producer.stop()
            self.frame_producer.wait(3000)

        # 停止系统监控
        self.system_monitor_timer.stop()

        event.accept()