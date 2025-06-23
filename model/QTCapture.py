# coding=utf-8
"""
    接口,提供给QT访问
    @project: EGGRECORDQT
    @Author：wjt
    @file： interface.py
    @date：2024/5/19 16:56
"""
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
from PyQt5.QtGui import QImage
import threading
import time
import os
import datetime


class CaptureInterface(QThread):
    frame_generated = pyqtSignal(object)

    def __init__(self, cfg):
        super().__init__()
        self.video = cfg['video']
        self.picture_save_path = cfg['picture_save_path']
        self.cap = None
        self.run_flag = True
        self.cap_false_count = 0
        self.paused = threading.Event()
        self.paused.set()  # 开始时不暂停
        self.width = cfg['width']
        self.height = cfg['height']

    def stop(self):
        self.run_flag = False
        self.resume()  # 确保在停止线程前恢复线程，防止线程阻塞在 self.paused.wait()

    def pause(self):
        self.paused.clear()  # 将事件状态设置为未触发，暂停线程

    def resume(self):
        self.paused.set()  # 将事件状态设置为触发，恢复线程

    def init_camera(self):
        print("初始化摄像头")
        self.cap = cv2.VideoCapture(self.video)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        if not self.cap.isOpened():
            print("摄像头初始化失败")
            time.sleep(2)

    def handle_capture_failure(self):
        self.cap_false_count += 1
        if self.cap_false_count > 3:
            print("摄像头获取失败，重启摄像头")
            self.release_camera()
            self.init_camera()

    def release_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def run(self):
        print("初始化QI采集接口")
        while self.run_flag:
            self.paused.wait()
            if self.cap is None:
                self.init_camera()
            ret, frame = self.cap.read()
            if not ret:
                self.handle_capture_failure()
                continue
            else:
                self.cap_false_count = 0
                cv2.imwrite(self.generate_path(), frame)
                # 将OpenCV帧转换为PyQt图像
                # image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create QImage
                image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
                self.frame_generated.emit(image)

        print("QI采集接口退出")

    # 用于生成图片保存路径
    def generate_path(self):
        # 作为文件名
        date = time.strftime('%Y%m%d', time.localtime(time.time()))
        pic_path = os.path.join(self.picture_save_path, date)
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        pic_name = datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S %f")
        pic_name = pic_name + '.jpg'
        pic_save_path = os.path.join(pic_path, pic_name)
        return pic_save_path
