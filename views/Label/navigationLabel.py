# coding=utf-8
"""
    导航界面：两个按钮（开始巡航、返回充电点）居中显示
    - 开始巡航（按计时顺序发送指令）：
      初始化 → 按 1、2、3、5、6 列的新 cage 点位顺序巡航，最后返回 initPoint
    - 返回充电点：initPoint
    到点确认：本版不依赖状态接口，严格按计时发送。
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QMessageBox, QTextEdit, QGridLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QIcon
import os
import time
import math
import logging

from model.communication.NavigateClient import NavigateClient
from model.utils.path_utils import resource_path, resolve_resource_path


def _res(name: str) -> str:
    return resource_path("resources", name)


def _find_config_path() -> str:
    cwd_cfg = os.path.join(os.getcwd(), 'configs', 'config.yaml')
    if os.path.isfile(cwd_cfg):
        return cwd_cfg
    return resource_path('configs', 'config.yaml')


def _resolve_cfg_paths(cfg: dict) -> dict:
    """Resolve resource paths for packaged runs."""
    if not isinstance(cfg, dict):
        return cfg
    cfg['modelPath'] = resolve_resource_path(
        cfg.get('modelPath'),
        os.path.join('resources', 'best2.pt')
    )
    cfg['tracking_config'] = resolve_resource_path(
        cfg.get('tracking_config'),
        os.path.join('configs', 'ocsort.yaml')
    )
    qr_decode = cfg.get('qr_decode')
    if isinstance(qr_decode, dict):
        qr_decode['wechat_model_dir'] = resolve_resource_path(
            qr_decode.get('wechat_model_dir'),
            os.path.join('resources', 'wechat')
        )
    return cfg


class PatrolThread(QThread):
    logSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal()
    completedSignal = pyqtSignal()

    def __init__(self, base_url: str, map_name: str, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.map_name = map_name
        self._stop = False

    def stop(self):
        self._stop = True

    def _log(self, text: str):
        self.logSignal.emit(text)

    def run(self):
        # 使用配置中的地图名称，不再强制硬编码
        self._log(f"开始任务，使用地图: {self.map_name}")
        client = NavigateClient(self.base_url, self.map_name)

        # Step 1: 初始化
        if self._stop:
            self.finishedSignal.emit()
            return

        self._log("正在初始化位置 (initPoint)...")
        # 修正：根据需求，初始化时携带 map_name=wenshi_10 和 init_point_name=initPoint
        # 已经在 NavigateClient.initialize_directly_point 中封装了 map_name 参数
        code, data, raw = client.initialize_directly_point("initPoint")
        self._log(f"初始化返回: code={code}, data={data}")

        if code != 200 or not client.is_success_response(data):
            self._log("初始化失败，终止巡航。")
            self.finishedSignal.emit()
            return

        # 等待初始化后状态就绪 (可选，等待2秒)
        time.sleep(2)
        
        # 确保状态为空闲才开始发送第一个点
        if not self._wait_until_idle_or_target(client, None):
            self._log("初始化后等待空闲超时，终止巡航。")
            self.finishedSignal.emit()
            return

        # Step 2: 定义巡航路径
        sequence = [
            "cage1001", "cage1190", "cage1001",
            "cage2001", "cage2190", "cage2001",
            "cage3001", "cage3190", "cage3001",
            "cage5001", "cage5190", "cage5001",
            "cage6001", "cage6190", "cage6001",
            "initPoint"
        ]

        # Step 3: 执行巡航
        completed = True
        for target in sequence:
            if self._stop:
                completed = False
                break

            # 1. 发送导航指令
            self._log(f"导航前往: {target}")
            code, data, raw = client.navigate_point(target)
            print(f"[导航] Target: {target}, Code: {code}, Raw: {raw}")

            if code != 200:
                 self._log(f"请求失败: {target}, code={code}")

            # 2. 等待导航开始生效
            time.sleep(2)

            # 3. 轮询状态直到完全到达
            # 判据：(status=6 且 position_name=target) OR (status=2)
            if not self._wait_until_idle_or_target(client, target):
                self._log(f"等待到达 {target} 超时或被中断。")
                break

        client.navigate_set_idle()
        if completed and not self._stop:
            self.completedSignal.emit()
        self.finishedSignal.emit()

    def _wait_until_idle_or_target(self, client, target_name):
        """
        轮询状态，直到：
        1. status = 2 (SH_NAV_IDLE) -> 到达
        2. status = 6 (SH_NAV_REACHED) 且 position_name == target_name -> 到达
        3. status = 6 且 position_name != target_name -> 可能是上一个点的残留 -> 尝试 set_idle -> 继续等
        """
        while not self._stop:
            code, data, raw = client.navigate_status()
            if code == 200 and data and isinstance(data, dict):
                # 提取状态
                inner_data = data.get("data", {})
                status = inner_data.get("status")
                current_pos = inner_data.get("position_name", "")

                # 1. 状态为 2 -> 空闲 -> 视为成功
                if status == 2:
                    return True

                # 2. 状态为 6 -> 到达
                if status == 6:
                    # 如果 target_name 为 None (例如初始化后等待空闲)，或者当前位置与目标一致
                    if target_name is None or current_pos == target_name:
                        return True
                    client.navigate_set_idle()
                        
            time.sleep(0.5)
        return False


class SingleTargetThread(QThread):
    logSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal()

    def __init__(self, base_url: str, map_name: str, target: str, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.map_name = map_name
        self.target = target
        self._stop = False

    def stop(self):
        self._stop = True

    def _log(self, text: str):
        self.logSignal.emit(text)

    def run(self):
        self._log(f"开始单点任务，使用地图: {self.map_name}")
        client = NavigateClient(self.base_url, self.map_name)

        # ----------------------------------------------------------------------
        # 原逻辑：强制空闲 -> 轮询等待空闲 -> 发送导航
        # 根据用户需求，删除自动初始化逻辑，直接执行导航
        # ----------------------------------------------------------------------
        if self._stop:
             self.finishedSignal.emit()
             return
        
        # 1. 设置空闲状态
        self._log("正在设置导航空闲状态...")
        client.navigate_set_idle()
        
        # 2. 轮询直到确认为空闲 (Status=2)
        # 避免立即发送指令导致冲突
        if not self._wait_until_idle_or_target(client, None):
             self._log("等待空闲超时或中断，取消导航。")
             self.finishedSignal.emit()
             return
        
        # 3. 发送导航点指令
        self._log(f"发送导航指令 -> {self.target}")
        code, data, raw = client.navigate_point(self.target)
        self._log(f"指令返回: code={code}, data={data}")
        
        # 4. 等待到达目标点
        if not self._wait_until_idle_or_target(client, self.target):
            self._log(f"等待到达 {self.target} 超时或被中断。")
            self.finishedSignal.emit()
            return
        
        self.finishedSignal.emit()

    def _wait_until_idle_or_target(self, client, target_name):
        # 等待最多 10秒 确保状态变为 IDLE 或 目标已到达
        start_t = time.time()
        while not self._stop and (time.time() - start_t < 10):
            code, data, raw = client.navigate_status()
            if code == 200 and data and isinstance(data, dict):
                inner_data = data.get("data", {})
                status = inner_data.get("status")
                current_pos = inner_data.get("position_name", "")
                
                # 1. status=2 -> 空闲
                if status == 2:
                    return True
                # 2. status=6 且 pos=target -> 到达
                if status == 6:
                    if target_name is None or current_pos == target_name:
                        return True
                    else:
                        # 是残留状态，主动清空
                        client.navigate_set_idle()

            time.sleep(0.5)
        return False


class _StartCheckThread(QThread):
    """后台执行"设置空闲 + 检查位置 + 初始化"，避免阻塞 UI 线程。"""
    resultSignal = pyqtSignal(bool, str)   # (ok, message)
    logSignal = pyqtSignal(str)

    def __init__(self, base_url, map_name, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.map_name = map_name

    def run(self):
        client = NavigateClient(self.base_url, self.map_name)

        # 1. 设置空闲状态（超时缩短到 3 秒，失败了也继续）
        try:
            code, data, _ = client.navigate_set_idle(timeout=3)
            self.logSignal.emit(f"设置导航空闲状态: code={code}, data={data}")
        except Exception as e:
            self.logSignal.emit(f"设置空闲状态异常（忽略）: {e}")

        # 2. 检查机器人位置（每次超时 3 秒，重试 3 次）
        last_data = None
        pos_ok = False
        for i in range(3):
            try:
                code, data, _ = client.get_robot_position(timeout=3)
                last_data = data
                self.logSignal.emit(f"获取机器人位置[{i+1}/3]: code={code}, data={data}")
                if code == 200 and self._is_near_start_pos(data):
                    pos_ok = True
                    break
            except Exception as e:
                self.logSignal.emit(f"获取位置异常: {e}")
            if i < 2:
                import time; time.sleep(0.2)

        if not pos_ok:
            self.resultSignal.emit(False, f"机器人不在起始区域，未启动巡航。最后位置={last_data}")
            return

        # 3. 初始化位置
        self.logSignal.emit("开始巡航前初始化位置 (initPoint)...")
        try:
            code, data, _ = client.initialize_directly_point("initPoint", timeout=5)
            self.logSignal.emit(f"初始化返回: code={code}, data={data}")
            if code != 200 or not client.is_success_response(data):
                self.resultSignal.emit(False, "初始化失败，未启动巡航。")
                return
        except Exception as e:
            self.resultSignal.emit(False, f"初始化异常，未启动巡航: {e}")
            return

        self.resultSignal.emit(True, "导航状态检查通过，正在启动巡航...")

    def _is_near_start_pos(self, data):
        try:
            if not isinstance(data, dict):
                return False
            inner = data.get("data", {})
            pos = inner.get("position", {})
            wx = float(pos.get("wx"))
            wy = float(pos.get("wy"))
            yaw = float(pos.get("yaw"))
        except Exception:
            return False
        target_wx, target_wy, target_yaw = -0.7137, 4.9743, -3.0829
        return (abs(wx - target_wx) <= 0.5 and abs(wy - target_wy) <= 0.5 and
                abs(abs(yaw - target_yaw) % (2 * 3.14159) - 3.14159) <= 0.5)


class NavigationLabel(QWidget):
    backToDetectSignal = pyqtSignal()
    patrolCompletedSignal = pyqtSignal()

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.parent = parent
        self.cfg = cfg or {}
        
        # 强制重新读取配置文件，确保最新的修改生效
        try:
            import yaml
            config_path = _find_config_path()
            with open(config_path, 'r', encoding='utf-8') as f:
                new_cfg = yaml.safe_load(f)
                if new_cfg:
                    self.cfg.update(new_cfg)
            _resolve_cfg_paths(self.cfg)
        except Exception as e:
            print(f"Warning: Failed to reload config.yaml: {e}")

        self.base_url = self.cfg.get('nav_base_url', "http://192.168.22.126:18000")
        # 默认值改为 wenshi_10，以防配置读取失败回退到 origin
        self.map_name = self.cfg.get('nav_map_name', "wenshi_10")

        self.patrol_thread = None
        self.single_thread = None

        self._initUI()

    def _initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部栏：左上角返回巡检界面按钮 + 居中标题
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setSpacing(8)

        back_btn = QPushButton(" 返回巡检界面")
        back_btn.setIcon(QIcon(_res("logout.png")))
        back_btn.setFixedHeight(36)
        back_btn.clicked.connect(self._on_back_to_detect)
        back_btn.setStyleSheet(
            """
            QPushButton { background-color: #FFFFFF; color: #353F5E; border: 1px solid #E5E7EB; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background-color: #F6F9FF; }
            """
        )

        title = QLabel("导航控制")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; background-color: #FFFFFF;")

        top_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        top_layout.addStretch(1)
        top_layout.addWidget(title, alignment=Qt.AlignCenter)
        top_layout.addStretch(2)

        top_bar.setStyleSheet("background-color: #FFFFFF;")
        main_layout.addWidget(top_bar)

        # 中部内容区 (包含所有按钮)
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignCenter)
        center_layout.setSpacing(30)
        center_layout.setContentsMargins(40, 40, 40, 40)

        # 1. 顶部大按钮区 (开始巡航、返回充电点)
        big_btns_layout = QHBoxLayout()
        big_btns_layout.setSpacing(40)
        big_btns_layout.setAlignment(Qt.AlignCenter)

        self.btn_start = QPushButton("开始巡航")
        self.btn_start.setIcon(QIcon(_res("start.png")))
        self.btn_start.setFixedSize(300, 100)  # 增大尺寸
        self.btn_start.clicked.connect(self.on_start_patrol)

        self.btn_return = QPushButton("返回充电点")
        self.btn_return.setIcon(QIcon(_res("power.png")))
        self.btn_return.setFixedSize(300, 100) # 增大尺寸
        self.btn_return.clicked.connect(lambda: self.on_single_navigate("initPoint", "返回充电点"))

        for b in (self.btn_start, self.btn_return):
            b.setStyleSheet(
                """
                QPushButton {
                    background-color: #1769FF; color: #FFFFFF; border: none; border-radius: 12px; font-size: 24px; font-weight: bold;
                }
                QPushButton:hover { background-color: #3A83FF; }
                QPushButton:pressed { background-color: #0F5BEF; }
                """
            )

        big_btns_layout.addWidget(self.btn_start)
        big_btns_layout.addWidget(self.btn_return)
        
        center_layout.addLayout(big_btns_layout)

        # 2. 列头/列尾 按钮区（按需求隐藏，仅保留开始巡航/返回充电点）
        show_nav_grid = False
        if show_nav_grid:
            # 尺寸放大 1.5倍: 160x60 -> 240x90
            grid_widget = QWidget()
            grid_layout = QGridLayout(grid_widget)
            grid_layout.setSpacing(15) # 调整间距

            # 定义按钮生成辅助函数
            def create_nav_btn(text, target, enabled=True):
                btn = QPushButton(text)
                # 调整尺寸：高度增大1.5倍 (60 * 1.5 = 90)，宽度适当增加以适配大字体 (160 -> 180)
                btn.setFixedSize(180, 90)

                style_normal = """
                    QPushButton {
                        background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB; border-radius: 8px; font-size: 20px; font-weight: bold;
                    }
                    QPushButton:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
                    QPushButton:pressed { background-color: #D1D5DB; }
                """

                style_disabled = """
                    QPushButton {
                        background-color: #E5E7EB; color: #9CA3AF; border: 1px solid #D1D5DB; border-radius: 8px; font-size: 20px; font-weight: bold;
                    }
                """

                btn.setStyleSheet(style_normal if enabled else style_disabled)

                if enabled:
                    btn.clicked.connect(lambda: self.on_single_navigate(target, f"导航至 {text}"))
                return btn

            # 第一排：1, 2, 3, 5, 6 列头
            grid_layout.addWidget(create_nav_btn("1列头 (1001)", "cage1001"), 0, 0)
            grid_layout.addWidget(create_nav_btn("2列头 (2001)", "cage2001"), 0, 1)
            grid_layout.addWidget(create_nav_btn("3列头 (3001)", "cage3001"), 0, 2)
            grid_layout.addWidget(create_nav_btn("5列头 (5001)", "cage5001"), 0, 3)
            grid_layout.addWidget(create_nav_btn("6列头 (6001)", "cage6001"), 0, 4)

            # 第二排：1, 2, 3, 5, 6 列尾
            grid_layout.addWidget(create_nav_btn("1列尾 (1190)", "cage1190"), 1, 0)
            grid_layout.addWidget(create_nav_btn("2列尾 (2190)", "cage2190"), 1, 1)
            grid_layout.addWidget(create_nav_btn("3列尾 (3190)", "cage3190"), 1, 2)
            grid_layout.addWidget(create_nav_btn("5列尾 (5190)", "cage5190"), 1, 3)
            grid_layout.addWidget(create_nav_btn("6列尾 (6190)", "cage6190"), 1, 4)

            # 居中显示 Grid
            hbox_grid = QHBoxLayout()
            hbox_grid.addStretch()
            hbox_grid.addWidget(grid_widget)
            hbox_grid.addStretch()

            center_layout.addLayout(hbox_grid)

        main_layout.addStretch(1)
        main_layout.addWidget(center_widget)
        main_layout.addStretch(2)

        # 底部日志
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(150)
        self.log_view.setStyleSheet("background-color: #FFFFFF; border-top: 1px solid #E5E7EB;")
        main_layout.addWidget(self.log_view)

    def append_log(self, text: str):
        self.log_view.append(text)
        # 滚动到底部
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_back_to_detect(self):
        try:
            self.backToDetectSignal.emit()
        except Exception as e:
            print(f"返回巡检界面信号发送失败: {e}")

    @pyqtSlot()
    def on_start_patrol(self):
        if self.patrol_thread and self.patrol_thread.isRunning():
            self.append_log("巡航任务已在执行中，忽略重复启动。")
            return

        # 禁用按钮，防止重复点击
        self.btn_start.setEnabled(False)
        self.append_log("正在检查导航状态，请稍候...")

        # 将所有阻塞网络操作放到后台线程，避免 UI 卡死
        self._start_check_thread = _StartCheckThread(self.base_url, self.map_name, self)
        self._start_check_thread.resultSignal.connect(self._on_start_check_result)
        self._start_check_thread.logSignal.connect(self.append_log)
        self._start_check_thread.start()

    def _on_start_check_result(self, ok: bool, msg: str):
        """后台检查完成后，在 UI 线程里做最终判断"""
        self.btn_start.setEnabled(True)
        self.append_log(msg)
        if not ok:
            QMessageBox.information(self, "提示", msg)
            return

        # 检查通过，启动巡航线程
        self._force_stop_threads()
        self.patrol_thread = PatrolThread(self.base_url, self.map_name, self)
        self.patrol_thread.logSignal.connect(self.append_log)
        self.patrol_thread.completedSignal.connect(self._on_patrol_completed)
        self.patrol_thread.finishedSignal.connect(lambda: self.append_log("巡航任务结束"))
        self.append_log("开始巡航：[wenshi_10] Init->(1,2,3,5,6列)->InitPoint")
        self.patrol_thread.start()

    def _on_patrol_completed(self):
        self.append_log("巡航任务已完成，准备结束检测。")
        self.patrolCompletedSignal.emit()

    @pyqtSlot()
    def on_single_navigate(self, target, desc):
        # 强制打断当前正在进行的巡航或导航任务
        self._force_stop_threads()
        
        # 启动新的单点导航任务
        self.single_thread = SingleTargetThread(self.base_url, self.map_name, target, self)
        self.single_thread.logSignal.connect(self.append_log)
        self.single_thread.finishedSignal.connect(lambda: self.append_log(f"{desc} 指令发送完成"))
        self.append_log(f"执行单点导航: {desc} -> {target}")
        self.single_thread.start()

    def _force_stop_threads(self):
        """强制停止所有正在运行的导航线程"""
        client = NavigateClient(self.base_url, self.map_name)

        if self.patrol_thread and self.patrol_thread.isRunning():
            self.patrol_thread.stop()
            self.patrol_thread.wait(500) # 等待一小会儿，不强求完全退出
            self.append_log("当前巡航任务已手动终止")
            
            # 停止后确保导航空闲
            client.navigate_set_idle()
            self.append_log("已设置导航空闲状态")

        if self.single_thread and self.single_thread.isRunning():
            self.single_thread.stop()
            self.single_thread.wait(500)
            self.append_log("上一个单点任务已终止")
            
            # 停止后确保导航空闲
            client.navigate_set_idle()
            self.append_log("已设置导航空闲状态")

    def _angle_diff(self, a: float, b: float) -> float:
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

    def _is_near_start_pos(self, data: dict) -> bool:
        """判断机器人是否在指定起始区域附近（位置误差<=0.5，角度误差<=0.5）"""
        try:
            if not isinstance(data, dict):
                return False
            inner = data.get("data", {})
            pos = inner.get("position", {})
            wx = float(pos.get("wx"))
            wy = float(pos.get("wy"))
            yaw = float(pos.get("yaw"))
        except Exception:
            return False

        target_wx = -0.7137
        target_wy = 4.9743
        target_yaw = -3.0829
        pos_tol = 0.5
        yaw_tol = 0.5
        return (abs(wx - target_wx) <= pos_tol and
                abs(wy - target_wy) <= pos_tol and
                self._angle_diff(yaw, target_yaw) <= yaw_tol)

    def _check_start_pos_with_retry(self, client, attempts: int = 5, interval_s: float = 0.2):
        last_data = None
        for i in range(attempts):
            code, data, raw = client.get_robot_position()
            last_data = data
            self.append_log(f"获取机器人位置[{i + 1}/{attempts}]: code={code}, data={data}")
            if code == 200 and self._is_near_start_pos(data):
                return True, data
            if i < attempts - 1:
                time.sleep(interval_s)
        return False, last_data

    def _is_busy(self):
        # 仅供 开始巡航 按钮使用，单点导航不再受此限制
        if self.patrol_thread and self.patrol_thread.isRunning():
            QMessageBox.information(self, "提示", "巡航任务正在执行中，请先停止或等待结束")
            return True
        if self.single_thread and self.single_thread.isRunning():
            QMessageBox.information(self, "提示", "正在执行单点导航指令，请稍候")
            return True
        return False

    @pyqtSlot()
    def close_threads(self):
        client = NavigateClient(self.base_url, self.map_name)
        try:
            if self.patrol_thread and self.patrol_thread.isRunning():
                self.patrol_thread.stop()
                self.patrol_thread.wait(3000)
        except Exception:
            pass
        try:
            if self.single_thread and self.single_thread.isRunning():
                self.single_thread.stop()
                self.single_thread.wait(3000)
        except Exception:
            pass
        try:
            client.navigate_set_idle()
        except Exception:
            pass

    def closeEvent(self, event):
        self.close_threads()
        event.accept()
