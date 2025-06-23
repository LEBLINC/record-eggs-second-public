# coding=utf-8
"""
    部署版本，去除二维码辅助推断
    @project: EGGRECORDQT
    @Author：wjt
    @file： main.py
    @date：2024/1/10 19:12
"""
import cv2
import time
from model.track.matchUtils import *
from pyzbar.pyzbar import decode
from collections import deque
import os


class MatchingCounting:
    def __init__(self, cfg):

        self.qr_dist = {}  # 用于记录二维码信息，键值位目标跟踪id
        self.qr_id = 1  # 代表记录的二维码数量
        self.egg_dist = {}  # 用于记录蛋信息，键值位目标跟踪id
        self.egg_id = 1  # 代表记录的蛋数量

        self.count = 0

        self.detector = cv2.wechat_qrcode_WeChatQRCode()
        self.edge_threshold = int(cfg['width'] * 0.05)
        self.egg_diff_num = 0
        self.qr_diff_num = 0
        self.width = int(cfg['width'] * 0.05)

        # 初始化队列
        self.egg_appear_nums_queue = deque(maxlen=10)
        self.qr_appear_nums_queue = deque(maxlen=10)

        self.match_center = cfg['width'] // 2
        self.match_range = cfg['width'] // 2 - self.edge_threshold

        self.picture_recognition_path = cfg['picture_recognition_path']
        if not os.path.exists(self.picture_recognition_path):
            os.makedirs(self.picture_recognition_path)

        self.color_map = {}

    def match(self, results, frame, draw_flag=True):
        egg_current_detects = {}
        qr_current_detects = []

        height, width = frame.shape[:2]

        if width != self.width:
            self.match_center = width // 2
            self.match_range = width // 2 - self.edge_threshold
            self.edge_threshold = int(width * 0.05)
            self.width = width

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = unpack_results(results)

        # 先预处理图片中所有的二维码信息
        for index, (box, track_id) in enumerate(zip(qr_boxes, qr_track_ids)):
            qr_current_detect = self._process_qr_codes(frame, track_id, box, width)
            qr_current_detects.append(qr_current_detect)

        # 预处理图片中所有的鸭蛋信息，以蛋为核心
        for index, (box, track_id) in enumerate(zip(egg_boxes, egg_track_ids)):
            egg_current_detects[track_id] = box
            self._process_egg_detection(box, track_id, qr_current_detects)

        # 查找最小平均距离进行蛋二维码匹配
        for box, track_id in zip(egg_boxes, egg_track_ids):
            if bool(self.egg_dist[track_id]['qr_dist']):
                # 定义匿名函数，按照 'distance' 键的值进行比较, 找到具有最小距离的二维码键
                min_key = min(self.egg_dist[track_id]['qr_dist'],
                              key=lambda key: self.egg_dist[track_id]['qr_dist'][key]['distance'])
                # 更新二维码中的匹配蛋数据
                if self.egg_dist[track_id]['min_qr_track_id'] and self.egg_dist[track_id]['min_qr_track_id'] != min_key:
                    del self.qr_dist[self.egg_dist[track_id]['min_qr_track_id']]['egg_dist'][track_id]
                else:
                    self.qr_dist[min_key]['egg_dist'][track_id] = track_id
                # 更新蛋中二维码的匹配记录
                self.egg_dist[track_id]['min_qr_track_id'] = min_key
                self.egg_dist[track_id]['record_time'] = time.time()

        # 更新二维码对应的eggBox，上传以二维码为核心
        for qr_track_id, qr_info in self.qr_dist.items():
            if qr_info['flag']:
                self.qr_dist[qr_track_id]['egg_boxs'] = egg_current_detects
                self.qr_dist[qr_track_id]['flag'] = False

        if draw_flag:
            for index, (box, track_id) in enumerate(zip(qr_boxes, qr_track_ids)):
                self._draw_rectangle(box, frame, 'qr', track_id)

            for index, (box, track_id) in enumerate(zip(egg_boxes, egg_track_ids)):
                self._draw_rectangle(box, frame, 'egg', track_id)
                if bool(self.egg_dist[track_id]['qr_dist']) and self.egg_dist[track_id]['min_qr_track_id'] is not None:
                    self._draw_lines(track_id, qr_track_ids, qr_boxes, box, frame)

            if len(self.qr_appear_nums_queue) != 0:
                qr_appear_nums = [self.qr_dist[temp_track_id]['appear_num'] for temp_track_id in
                                  self.qr_appear_nums_queue if temp_track_id in self.qr_dist.keys()]
                if len(qr_appear_nums) > 0:
                    qr_mean = np.mean(qr_appear_nums)
                    self.qr_diff_num = qr_mean / 2

            if len(self.egg_appear_nums_queue) != 0:
                egg_appear_nums = [self.egg_dist[temp_track_id]['appear_num'] for temp_track_id in
                                   self.egg_appear_nums_queue if temp_track_id in self.egg_dist.keys()]
                if len(egg_appear_nums) > 0:
                    egg_mean = np.mean(egg_appear_nums)
                    self.egg_diff_num = egg_mean / 1.5
            # 绘制左边缘线
            cv2.line(frame, (self.edge_threshold, 0), (self.edge_threshold, height), (0, 0, 255, 128), 1,
                     lineType=cv2.LINE_AA)

            # 绘制右边缘线
            cv2.line(frame, (width - self.edge_threshold, 0), (width - self.edge_threshold, height),
                     (0, 0, 255, 128),
                     1,
                     lineType=cv2.LINE_AA)

    def _process_qr_codes(self, frame, track_id, box, width):
        """
        :param width:
        :param frame:
        :param track_id: 跟踪过程的id，汇集了蛋与二维码
        :param box: XYXY样式的矩形框坐标
        :return: 用于计算二维码变形程度与识别二维码
        """
        if track_id not in self.qr_dist.keys():
            # 初始化二维码记录
            self.qr_dist[track_id] = {
                'qr_id': self.qr_id,
                'record_time': time.time(),
                'egg_num': 0,  # 对应在该二维码下蛋的数量
                'egg_track_ids': [],
                'flag': True,  # 匹配更新判断
                'frame': None,
                'qr_box': None,
                'min_mid': 10000,
                'egg_boxs': None,
                'aspect_ratio': None,
                'appear_num': 1,  # 该二维码出现次数
                'decode_id': None,
                'diff_num': self.qr_diff_num,
                'count': self.count,
                'egg_dist': {},
            }
            self.qr_id += 1
            # 为每个 track_id 生成随机的背景颜色
            self.color_map[track_id] = generate_random_color()
            self.qr_appear_nums_queue.append(track_id)
        else:
            # 更新二维码记录
            self.qr_dist[track_id]['record_time'] = time.time()
            self.qr_dist[track_id]['appear_num'] += 1
            self.qr_dist[track_id]['diff_num'] = self.qr_diff_num
            self.qr_dist[track_id]['count'] = self.count

        mid = calculate_mid(box, width)
        # 计算并更新图片形变程度
        aspect_ratio = calculate_aspect_ratio(box)
        self.qr_dist[track_id]['aspect_ratio'] = aspect_ratio

        qr_current_detect = {
            'box': box,
            'qr_id': self.qr_dist[track_id]['qr_id'],
            'track_id': track_id,
            'aspect_ratio': aspect_ratio,
            'mid': mid
        }

        # 用于保存上传图像
        if mid < self.qr_dist[track_id]['min_mid']:
            self.qr_dist[track_id]['min_mid'] = mid
            self.qr_dist[track_id]['frame'] = frame.copy()
            self.qr_dist[track_id]['qr_box'] = box
            self.qr_dist[track_id]['flag'] = True

        # 判断二维码是否识别，未识别则使用zbar与微信库进行识别
        if self.qr_dist[track_id]['decode_id'] is None:
            x1, y1, x2, y2 = adjust_bounds(box)
            # 截取图像
            cropped_image = frame[y1:y2, x1:x2]
            qr_codes = decode(cropped_image)
            if len(qr_codes) == 1:
                self.qr_dist[track_id]['decode_id'] = qr_codes[0].data.decode('utf-8')
            # else:
            #     res, points = self.detector.detectAndDecode(cropped_image)
            #     if len(res) == 1:
            #         self.qr_dist[track_id]['decode_id'] = int(res[0])

        return qr_current_detect

    def _process_egg_detection(self, box, track_id, qr_current_detects):
        """
        :param box: XYXY样式的矩形框坐标
        :param track_id: 跟踪过程中的id
        :param qr_current_detects: 当前帧检测到的二维码信息
        :return: 用于处理蛋数据，与蛋的上一次匹配到的所有二维码进行计算，更新匹配
        """
        if track_id not in self.egg_dist.keys():
            # 初始化蛋检测记录
            self.egg_dist[track_id] = {
                'egg_id': self.egg_id,
                'qr_dist': {},
                'min_qr_track_id': None,
                'record_time': time.time(),
                'appear_num': 1,  # 该蛋出现次数
                'diff_num': self.egg_diff_num,
                'count': self.count
            }
            self.color_map[track_id] = generate_random_color()
            self.egg_id += 1
            self.egg_appear_nums_queue.append(track_id)
        else:
            # 更新最新记录
            self.egg_dist[track_id]['record_time'] = time.time()
            self.egg_dist[track_id]['appear_num'] += 1
            self.egg_dist[track_id]['diff_num'] = self.egg_diff_num
            self.egg_dist[track_id]['count'] = self.count

        egg_qr_dist = self.egg_dist[track_id]['qr_dist']
        x1_1, y1_1, x2_1, y2_1 = box[0], box[1], box[2], box[3]
        center_x1 = (x1_1 + x2_1) / 2
        if center_x1 < self.match_center - self.match_range or center_x1 > self.match_center + self.match_range:
            return

        for i, qr_detect in enumerate(qr_current_detects):
            mid = qr_detect['mid']
            qr_box = qr_detect['box']
            qr_id = qr_detect['qr_id']
            qr_track_id = qr_detect['track_id']

            distance = calculate_center_distance(box, qr_box)

            # 二维码id在蛋的匹配记录里
            if qr_track_id in egg_qr_dist.keys():
                distances = self.egg_dist[track_id]['qr_dist'][qr_track_id]['aspect_ratio']
                distances.append(distance)
                self.egg_dist[track_id]['qr_dist'][qr_track_id]['distance'] = np.mean(distances)
            else:
                self.egg_dist[track_id]['qr_dist'][qr_track_id] = {
                    'aspect_ratio': [distance],
                    'mid': mid,
                    'distance': distance,
                    'qr_id': qr_id
                }

    def update_and_delete_records(self):
        """
        :return: 用于汇集蛋当前帧匹配的结果，蛋大于10秒则将该蛋与二维码做最终匹配记录，后删除
        """
        del_qr_ids = []
        del_egg_ids = []
        result = []
        self.count += 1

        for egg_track_id, egg_info in self.egg_dist.items():
            min_qr_track_id = self.egg_dist[egg_track_id]['min_qr_track_id']  # 蛋对于的二维码id
            if self.count - egg_info['count'] > 30:
                appear_num = self.egg_dist[egg_track_id]['appear_num']
                diff_num = self.egg_dist[egg_track_id]['diff_num']
                # 删除在二维码匹配记录中出现次数较少的记录
                if min_qr_track_id in self.qr_dist.keys() and appear_num < diff_num:
                    del self.qr_dist[min_qr_track_id]['egg_dist'][egg_track_id]
                del_egg_ids.append(egg_track_id)

        for qr_track_id, qr_info in self.qr_dist.items():
            flag = True
            for egg_track_id in self.qr_dist[qr_track_id]['egg_dist'].keys():
                if egg_track_id in self.egg_dist.keys():
                    flag = False
            if self.count - qr_info['count'] > 30 and flag:
                egg_num = len(self.qr_dist[qr_track_id]['egg_dist'])
                record_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                cage_id = self.qr_dist[qr_track_id]['decode_id']
                qr_box = self.qr_dist[qr_track_id]['qr_box']
                cv2.rectangle(self.qr_dist[qr_track_id]['frame'], (qr_box[0], qr_box[1]), (qr_box[2], qr_box[3]),
                              (0, 255, 0), 2)
                text = f"QR:{cage_id}"
                cv2.putText(
                    self.qr_dist[qr_track_id]['frame'],
                    text,
                    (qr_box[0] - 10, qr_box[1] - 10),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                for egg_track_id, egg_box in self.qr_dist[qr_track_id]['egg_boxs'].items():
                    if egg_track_id in self.qr_dist[qr_track_id]['egg_dist'].keys():
                        cv2.rectangle(self.qr_dist[qr_track_id]['frame'], (egg_box[0], egg_box[1]),
                                      (egg_box[2], egg_box[3]), (0, 255, 0), 2)
                        text = f"egg"
                        cv2.putText(
                            self.qr_dist[qr_track_id]['frame'],
                            text,
                            (egg_box[0] - 10, egg_box[1] - 10),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )
                        center1 = ((qr_box[0] + qr_box[2]) // 2, (qr_box[1] + qr_box[3]) // 2)
                        center2 = ((egg_box[0] + egg_box[2]) // 2, (egg_box[1] + egg_box[3]) // 2)
                        # 在图像上绘制连接线
                        cv2.line(self.qr_dist[qr_track_id]['frame'], center1, center2, (0, 0, 255), 4)

                appear_num = self.qr_dist[qr_track_id]['appear_num']
                diff_num = self.qr_dist[qr_track_id]['diff_num']
                if cage_id is not None and appear_num > diff_num:
                    frame_path = os.path.join(self.picture_recognition_path,
                                              str(cage_id) + '_' + str(int(time.time() * 1000)) + '.jpg')
                    cv2.imwrite(frame_path, self.qr_dist[qr_track_id]['frame'])
                    send_data = {
                        'cage_id': cage_id,
                        'record_time': record_time,
                        'egg_num': egg_num,
                        'track_id': qr_track_id,
                        'appear_num': appear_num,
                        'frame_path': frame_path
                    }
                    result.append(send_data)
                    del_qr_ids.append(qr_track_id)

        # 删除大于10秒的条目
        for del_egg_id in del_egg_ids:
            del self.egg_dist[del_egg_id]
        # 删除大于10秒的条目
        for del_qr_id in del_qr_ids:
            del self.qr_dist[del_qr_id]

        return result

    def _draw_lines(self, track_id, track_ids, boxes, box, frame):
        index = find_index_of_id(self.egg_dist[track_id]['min_qr_track_id'], track_ids)
        if index != -1:
            # 提取矩形框的中心点坐标
            center1 = (
                (boxes[index][0] + boxes[index][2]) // 2, (boxes[index][1] + boxes[index][3]) // 2)
            center2 = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            # 在图像上绘制连接线
            cv2.line(frame, center1, center2, (0, 0, 255), 4)

    def _draw_rectangle(self, box, frame, name, track_id):
        if name == 'qr':
            if self.qr_dist[track_id]['decode_id'] is not None:
                str_display = 'Identify: ' + str(self.qr_dist[track_id]['decode_id'])
            else:
                str_display = 'wait: ' + str(self.qr_dist[track_id]['qr_id'])
        else:
            str_display = ''

        # 获取 track_id 对应的背景颜色
        bg_color = self.color_map.get(track_id, (0, 0, 0))  # 默认为黑色背景

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{name}{str_display}"

        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)

        # 计算背景矩形的位置
        bg_rect_x = box[0] - 10
        bg_rect_y = box[1] - 35
        bg_rect_width = text_width + 5
        bg_rect_height = text_height + 10

        # 绘制背景矩形
        cv2.rectangle(frame, (bg_rect_x, bg_rect_y), (bg_rect_x + bg_rect_width, bg_rect_y + bg_rect_height), bg_color,
                      -1)

        cv2.putText(
            frame,
            text,
            (box[0] - 10, box[1] - 10),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (255, 255, 255),
            2,
        )
