# coding=utf-8
"""
    接口,提供给QT访问
    @project: EGGRECORDQT
    @Author：wjt
    @file： interface.py
    @date：2024/5/19 16:56
"""
from PyQt5.QtCore import QThread, pyqtSignal
from model.track.yoloTrack import YOLOTrack
from model.match.matchingCounting import MatchingCounting
import cv2
import queue
from PyQt5.QtGui import QImage
import threading
import time
from model.communication.SendHttp import SendHttp
from model.utils.exception import exception_handler


class BaseThread(QThread):
    def __init__(self):
        super().__init__()
        self.run_flag = True
        self.paused = threading.Event()
        self.paused.set()  # 开始时不暂停

    def stop(self):
        self.run_flag = False
        self.resume()  # 确保在停止线程前恢复线程，防止线程阻塞

    def pause(self):
        self.paused.clear()  # 暂停线程

    def resume(self):
        self.paused.set()  # 恢复线程


class FrameThread(BaseThread):
    def __init__(self, video, frame_queue, cfg):
        super().__init__()
        self.video = video
        self.cap = None
        self.frame_queue = frame_queue
        self.cap_false_count = 0
        self.width = cfg.get('width', 640)
        self.height = cfg.get('height', 480)

    @exception_handler
    def init_camera(self):
        print("初始化摄像头")
        self.cap = cv2.VideoCapture(self.video)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        if not self.cap.isOpened():
            print("摄像头初始化失败")
            time.sleep(2)

    @exception_handler
    def handle_capture_failure(self):
        self.cap_false_count += 1
        if self.cap_false_count > 3:
            print("摄像头获取失败，重启摄像头")
            self.release_camera()
            self.init_camera()

    @exception_handler
    def release_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    @exception_handler
    def run(self):
        print("FrameThread启动")
        while self.run_flag:
            self.paused.wait()
            if self.cap is None:
                self.init_camera()
            ret, frame = self.cap.read()
            if not ret:
                self.handle_capture_failure()
                continue
            self.cap_false_count = 0
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                print("Frame队列已满，丢弃帧")
        self.release_camera()
        print("FrameThread退出")


class TrackThread(BaseThread):
    def __init__(self, YOLOTrack, frame_queue, track_queue):
        super().__init__()
        self.YOLOTrack = YOLOTrack
        self.frame_queue = frame_queue
        self.track_queue = track_queue

    @exception_handler
    def run(self):
        print("TrackThread启动")
        while self.run_flag:
            self.paused.wait()
            try:
                frame = self.frame_queue.get(timeout=1)  # 使用timeout来避免无限阻塞
            except queue.Empty:
                continue
            track_results = self.YOLOTrack.track(frame)
            while self.run_flag:
                try:
                    self.track_queue.put((frame, track_results), timeout=1)
                    break
                except queue.Full:
                    continue
        print("TrackThread退出")


class MatchThread(BaseThread):
    def __init__(self, matchingCounting, track_queue, result_queue):
        super().__init__()
        self.matchingCounting = matchingCounting
        self.track_queue = track_queue
        self.result_queue = result_queue

    @exception_handler
    def run(self):
        print("MatchThread启动")
        while self.run_flag:
            self.paused.wait()
            try:
                frame, track_results = self.track_queue.get(timeout=1)  # 使用timeout来避免无限阻塞
            except queue.Empty:
                continue
            if track_results[0].boxes.id is not None:
                self.matchingCounting.match(track_results, frame)
            match_results = self.matchingCounting.update_and_delete_records()
            self.result_queue.put((frame, match_results), timeout=1)
        print("MatchThread退出")


class HTTPThread(BaseThread):
    def __init__(self, no_picture_result_queue, cfg):
        super().__init__()
        self.no_picture_result_queue = no_picture_result_queue
        self.SendHttp = SendHttp(cfg)

    @exception_handler
    def run(self):
        print("HTTPThread启动")
        while self.run_flag:
            self.paused.wait()
            try:
                no_picture_result = self.no_picture_result_queue.get(timeout=1)
                self.SendHttp.http_post(no_picture_result)
            except queue.Empty:
                continue
        print("HTTPThread退出")


class Interface(QThread):
    frame_generated = pyqtSignal(object)

    def __init__(self, cfg):
        super().__init__()
        self.YOLOTrack = YOLOTrack(cfg)
        print(cfg)
        self.matchingCounting = MatchingCounting(cfg)
        self.frame_queue = queue.Queue(maxsize=3)
        self.track_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue()
        self.no_picture_result_queue = queue.Queue()
        self.run_flag = True
        self.paused = threading.Event()
        self.paused.set()  # 开始时不暂停
        video = cfg['video']
        if cfg['mode'] == 2:
            video = cfg['demo_video']
        self.frame_thread = FrameThread(video, self.frame_queue, cfg)
        self.track_thread = TrackThread(self.YOLOTrack, self.frame_queue, self.track_queue)
        self.match_thread = MatchThread(self.matchingCounting, self.track_queue, self.result_queue)
        self.http_thread = HTTPThread(self.no_picture_result_queue, cfg)

    def _start_interface(self):
        self.frame_thread.start()
        self.track_thread.start()
        self.match_thread.start()
        self.http_thread.start()

    def _stop_interface(self):
        self.frame_thread.stop()
        self.track_thread.stop()
        self.match_thread.stop()
        self.http_thread.stop()
        self.frame_thread.wait()
        self.track_thread.wait()
        self.match_thread.wait()
        self.http_thread.wait()

    def _resume_interface(self):
        self.frame_thread.resume()
        self.track_thread.resume()
        self.match_thread.resume()
        self.http_thread.resume()

    def _pause_interface(self):
        self.frame_thread.pause()
        self.track_thread.pause()
        self.match_thread.pause()
        self.http_thread.pause()

    def stop(self):
        self.run_flag = False
        self._stop_interface()
        self.resume()  # 确保在停止线程前恢复线程，防止线程阻塞在 self.paused.wait()

    def pause(self):
        self.paused.clear()  # 将事件状态设置为未触发，暂停线程
        self._pause_interface()

    def resume(self):
        self.paused.set()  # 将事件状态设置为触发，恢复线程
        self._resume_interface()

    @exception_handler
    def run(self):
        print("初始化QI多线程检测接口")
        self._start_interface()
        while self.run_flag:
            self.paused.wait()
            try:
                frame, match_results = self.result_queue.get(timeout=1)  # 使用timeout来避免无限阻塞
                for match_result in match_results:
                    self.no_picture_result_queue.put(match_result, timeout=1)
            except queue.Empty:
                continue
            # 将OpenCV帧转换为PyQt图像
            # image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create QImage
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            # image = QImage(frame.data, frame.shape[1], frame.shape[0],  QImage.Format_BGR888)
            self.frame_generated.emit(image)
        print("QI多线程检测接口退出")
