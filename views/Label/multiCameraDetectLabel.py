# coding=utf-8
"""
    多摄像头检测界面（改进版）
    @project: EGGRECORDQT
    @Author：lzy
    @file： multiCameraDetectLabel.py
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
                             QLabel, QGroupBox, QGridLayout, QMessageBox,
                             QTextEdit, QSplitter, QProgressBar, QFrame, QComboBox, QScrollArea, QDialog, QApplication,
                             QStackedLayout, QToolTip, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QEvent, QPoint, pyqtProperty, QPropertyAnimation, QEasingCurve, QThread
from PyQt5.QtGui import QPixmap, QIcon, QFont, QFontMetrics, QPalette, QPainter, QColor, QBrush, QPen
import datetime
from model.MultiCameraInterface import MultiCameraInterface
import os
import psutil
import time
import re
from model.utils.path_utils import resource_path


def _res(name: str) -> str:
    return resource_path("resources", name)


class _ProducerInitThread(QThread):
    """后台线程：构造 MultiCameraInterface（加载模型、初始化摄像头），避免阻塞 UI。"""
    succeeded = pyqtSignal(object)   # 传出构造好的 MultiCameraInterface 实例
    failed = pyqtSignal(str)          # 传出错误信息

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config

    def run(self):
        try:
            producer = MultiCameraInterface(self._config)
            self.succeeded.emit(producer)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class CustomToolTip(QWidget):
    """自定义工具提示窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        # 设置布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 创建内容标签
        self.content_label = QLabel()
        self.content_label.setStyleSheet("""
            QLabel {
                background-color: #1769FF;
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 12px;
                font-weight: normal;
                border-radius: 6px;
                font-family: Arial, sans-serif;
            }
        """)
        self.content_label.setAlignment(Qt.AlignLeft)
        self.content_label.setTextFormat(Qt.RichText)  # 支持富文本
        self.layout.addWidget(self.content_label)

    def show_tooltip(self, text, position):
        """显示工具提示"""
        self.content_label.setText(text)
        self.adjustSize()
        self.move(position)
        self.show()
        self.raise_()

    def hide_tooltip(self):
        """隐藏工具提示"""
        self.hide()


class DetectionImageLabel(QLabel):
    """自定义图片标签，支持检测信息的工具提示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_info = None
        self.setMouseTracking(True)
        self.custom_tooltip = CustomToolTip()

    def set_detection_info(self, column_info, cage_id, egg_num):
        """设置检测信息"""
        self.detection_info = {
            'column_info': column_info,
            'cage_id': cage_id,
            'egg_num': egg_num
        }

    def enterEvent(self, event):
        """鼠标进入事件"""
        if self.detection_info:
            # 创建自定义工具提示内容
            column_info = self.detection_info['column_info']
            cage_id = self.detection_info['cage_id']
            egg_num = self.detection_info['egg_num']

            # 格式化工具提示文本，实现分两行显示和左对齐
            tooltip_text = f"""<span style='text-decoration: underline;'>{column_info}列  {cage_id}</span><br/>{egg_num}枚"""

            # 计算工具提示位置
            global_pos = self.mapToGlobal(QPoint(0, -50))  # 显示在图片上方

            # 显示自定义工具提示
            self.custom_tooltip.show_tooltip(tooltip_text, global_pos)

        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开事件"""
        # 隐藏自定义工具提示
        self.custom_tooltip.hide_tooltip()
        super().leaveEvent(event)

class AspectRatioLabel(QLabel):
    """保持固定宽高比的 QLabel（用于摄像头画面，避免“框很高但画面被迫居中留黑边”）"""

    def __init__(self, aspect_ratio=16 / 9, parent=None):
        super().__init__(parent)
        self._aspect_ratio = 16 / 9
        self.set_aspect_ratio(aspect_ratio)

        # 宽度可扩展，高度由 width -> heightForWidth 决定（更贴近实际画面比例）
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_aspect_ratio(self, aspect_ratio):
        try:
            r = float(aspect_ratio)
            if r > 0:
                self._aspect_ratio = r
                self.updateGeometry()
        except Exception:
            pass

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        try:
            if self._aspect_ratio > 0:
                return max(1, int(w / self._aspect_ratio))
        except Exception:
            pass
        return super().heightForWidth(w)


class CageProgressCell(QFrame):
    """
    单个笼位“进度条方块”控件：
    - 方块本身固定 square_px * square_px（例如 20px）
    - 用边框表示“当前扫描位置”（不额外增加尺寸）
    """

    def __init__(self, square_px=30, parent=None):
        super().__init__(parent)
        self.square_px = int(max(1, square_px))
        self._has_egg = False
        self._is_current = False

        self.setFrameShape(QFrame.NoFrame)
        self.setFixedSize(self.square_px, self.square_px)

        self._apply_style()

    def set_has_egg(self, has_egg: bool):
        has_egg = bool(has_egg)
        if self._has_egg == has_egg:
            return
        self._has_egg = has_egg
        self._apply_style()

    def set_is_current(self, is_current: bool):
        is_current = bool(is_current)
        if self._is_current == is_current:
            return
        self._is_current = is_current
        self._apply_style()

    def _apply_style(self):
        # 颜色：灰=无蛋，绿=有蛋（保持）
        fill = "#22C55E" if self._has_egg else "#D1D5DB"
        # 边框：蓝=当前扫描；灰=普通
        border_color = "#1769FF" if self._is_current else "#BFC5D0"
        self.setStyleSheet(f"background-color: {fill}; border: 1px solid {border_color}; border-radius: 2px;")


class CageProgressColumn(QWidget):
    """单列笼位进度条：120个方块、每页显示10个、扫描时自下往上推进，翻页时平滑滚动。"""

    def __init__(self, title: str, total_slots: int = 120, page_size: int = 10, square_px: int = 5, cell_gap: int = 40, parent=None):
        super().__init__(parent)
        self.total_slots = int(max(1, total_slots))
        self.page_size = int(max(1, min(page_size, self.total_slots)))
        self.square_px = int(max(1, square_px))

        # 单元尺寸与间距（可配置）
        self.cell_outer_px = self.square_px
        try:
            self.cell_gap = int(max(0, cell_gap))
        except Exception:
            self.cell_gap = 40  # 单元之间上下间距(px)

        self._current_idx = None
        self._current_group = 1
        self._scroll_anim = None  # 避免动画被GC

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self.title_label = QLabel(str(title))
        self.title_label.setAlignment(Qt.AlignCenter)
        # 标题：高度尽量贴合文字
        title_font = QFont(self.title_label.font())
        title_font.setPointSize(13)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #353F5E;")
        try:
            fm = QFontMetrics(title_font)
            self.title_label.setFixedHeight(fm.height() + 4)
        except Exception:
            self.title_label.setFixedHeight(20)

        self.group_label = QLabel("")
        self.group_label.setAlignment(Qt.AlignCenter)
        # 分组：两行显示，字号稍大，高度尽量贴合两行文字
        group_font = QFont(self.group_label.font())
        group_font.setPointSize(13)
        group_font.setBold(False)
        self.group_label.setFont(group_font)
        self.group_label.setWordWrap(True)
        self.group_label.setStyleSheet("color: #6F7A93;")
        try:
            gfm = QFontMetrics(group_font)
            self.group_label.setFixedHeight(gfm.height() * 2 + 6)
        except Exception:
            self.group_label.setFixedHeight(38)

        root.addWidget(self.title_label)
        root.addWidget(self.group_label)

        self.scroll = QScrollArea()
        # 让内容宽度随列宽变化，以便方块始终在每列水平居中
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("background: transparent;")
        # 让方块列在视口中水平居中（避免看起来靠左）
        try:
            self.scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        except Exception:
            pass

        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(self.cell_gap)
        self.container_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self.cells = {}  # index(1..N) -> CageProgressCell
        # 关键：按 120 -> 1 的顺序从上到下摆放，使得“从下往上”就是 1->120
        for idx in range(self.total_slots, 0, -1):
            cell = CageProgressCell(square_px=self.square_px, parent=self.container)
            self.cells[idx] = cell
            self.container_layout.addWidget(cell, 0, Qt.AlignHCenter)

        self.scroll.setWidget(self.container)

        # 固定视口高度：每页显示10个（含间距）
        visible_h = self.page_size * self.cell_outer_px + (self.page_size - 1) * self.cell_gap
        self.scroll.setFixedHeight(visible_h)
        root.addWidget(self.scroll)

        self.reset()

    def reset(self):
        self._current_idx = None
        self._current_group = 1
        for idx, cell in self.cells.items():
            cell.set_is_current(False)
            cell.set_has_egg(False)
        self._update_group_label(1)
        # 等布局计算后再滚动，否则最大值可能还是0
        QTimer.singleShot(0, lambda: self.scroll_to_group(1, animate=False))

    def mark_has_egg(self, idx: int):
        try:
            idx = int(idx)
        except Exception:
            return
        cell = self.cells.get(idx)
        if cell:
            cell.set_has_egg(True)

    def set_current_index(self, idx: int, animate: bool = True):
        try:
            idx = int(idx)
        except Exception:
            return
        if idx < 1 or idx > self.total_slots:
            return

        # 取消旧高亮
        if self._current_idx is not None and self._current_idx in self.cells:
            self.cells[self._current_idx].set_is_current(False)

        # 设置新高亮
        self._current_idx = idx
        if idx in self.cells:
            self.cells[idx].set_is_current(True)

        group = int((idx - 1) // self.page_size) + 1
        if group != self._current_group:
            self._current_group = group
            self._update_group_label(group)
            self.scroll_to_group(group, animate=bool(animate))
        else:
            self._update_group_label(group)

    def update_by_scan(self, idx: int, has_egg: bool, animate: bool = True):
        if bool(has_egg):
            self.mark_has_egg(idx)
        self.set_current_index(idx, animate=bool(animate))

    def _update_group_label(self, group: int):
        total_groups = int((self.total_slots - 1) // self.page_size) + 1
        start_idx = (group - 1) * self.page_size + 1
        end_idx = min(group * self.page_size, self.total_slots)
        # 两行显示（符合用户需求）
        self.group_label.setText(f"第{group}/{total_groups}组\n（{start_idx}-{end_idx}）")

    def scroll_to_group(self, group: int, animate: bool = True):
        """将视口滚动到指定组，使该组的“顶部方块”对齐到视口顶部。"""
        try:
            group = int(group)
        except Exception:
            return
        if group < 1:
            group = 1
        total_groups = int((self.total_slots - 1) // self.page_size) + 1
        if group > total_groups:
            group = total_groups

        end_idx = min(group * self.page_size, self.total_slots)  # 该组顶部（在视口顶部）
        target_cell = self.cells.get(end_idx)
        if not target_cell:
            return

        bar = self.scroll.verticalScrollBar()
        # 兜底：布局未完成时最大值可能为0，稍后再试
        if bar.maximum() == 0:
            QTimer.singleShot(50, lambda: self.scroll_to_group(group, animate=animate))
            return

        target_value = int(target_cell.y())
        target_value = max(bar.minimum(), min(bar.maximum(), target_value))

        if not animate:
            bar.setValue(target_value)
            return

        # 平滑滚动动画（像网页侧边滚动条一样）
        try:
            if self._scroll_anim is not None:
                self._scroll_anim.stop()
        except Exception:
            pass

        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(250)
        anim.setStartValue(bar.value())
        anim.setEndValue(target_value)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        self._scroll_anim = anim
        anim.start()


class CageProgressPanel(QWidget):
    """左侧“无画面模式”进度条面板：4路/6路（左侧N列 + 过道 + 右侧N列）。"""

    def __init__(
        self,
        camera_labels: list,
        total_slots: int,
        page_size: int,
        square_px: int,
        cell_gap: int,
        parent=None,
        left_camera_order=None,
        right_camera_order=None,
    ):
        super().__init__(parent)
        self._camera_to_column = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        def safe_label(i, fallback):
            try:
                if isinstance(camera_labels, list) and i < len(camera_labels):
                    return str(camera_labels[i])
            except Exception:
                pass
            return str(fallback)

        # 默认顺序：4路(左上/左下 | 过道 | 右下/右上)，6路(左上/左中/左下 | 过道 | 右上/右中/右下)
        total_cams = len(camera_labels) if isinstance(camera_labels, list) else 0
        if left_camera_order is None or right_camera_order is None:
            if total_cams <= 4:
                left_camera_order = list(range(min(2, total_cams)))
                right_camera_order = [3, 2] if total_cams >= 4 else list(range(2, total_cams))
            else:
                left_camera_order = list(range(min(3, total_cams)))
                right_camera_order = list(range(3, total_cams))

        total_cols = len(left_camera_order) + len(right_camera_order)
        layout.setSpacing(8 if total_cols > 4 else 12)

        left_fallback = ["左上", "左中", "左下"] if len(left_camera_order) >= 3 else ["左上", "左下"]
        right_fallback = ["右上", "右中", "右下"] if len(right_camera_order) >= 3 else ["右下", "右上"]

        left_columns = []
        for pos, cam_idx in enumerate(left_camera_order):
            fallback = left_fallback[pos] if pos < len(left_fallback) else f"左{pos + 1}"
            col = CageProgressColumn(safe_label(cam_idx, fallback), total_slots, page_size, square_px, cell_gap, self)
            left_columns.append(col)

        right_columns = []
        for pos, cam_idx in enumerate(right_camera_order):
            fallback = right_fallback[pos] if pos < len(right_fallback) else f"右{pos + 1}"
            col = CageProgressColumn(safe_label(cam_idx, fallback), total_slots, page_size, square_px, cell_gap, self)
            right_columns.append(col)

        aisle_widget = QFrame(self)
        aisle_widget.setFixedWidth(28 if total_cols > 4 else 36)
        aisle_widget.setStyleSheet(
            "border-left: 5px solid #D1D5DB; border-right: 5px solid #D1D5DB; background-color: #F0F2F5;")
        aisle_layout = QVBoxLayout(aisle_widget)
        aisle_layout.setContentsMargins(0, 0, 0, 0)
        aisle_label = QLabel("过\n道", aisle_widget)
        aisle_label.setAlignment(Qt.AlignCenter)
        aisle_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #909399; border: none;")
        aisle_layout.addStretch(1)
        aisle_layout.addWidget(aisle_label)
        aisle_layout.addStretch(1)

        for col in left_columns:
            layout.addWidget(col, 1)
        layout.addWidget(aisle_widget, 0)
        for col in right_columns:
            layout.addWidget(col, 1)

        for cam_idx, col in zip(left_camera_order, left_columns):
            self._camera_to_column[int(cam_idx)] = col
        for cam_idx, col in zip(right_camera_order, right_columns):
            self._camera_to_column[int(cam_idx)] = col

    def reset(self):
        for col in self._camera_to_column.values():
            try:
                col.reset()
            except Exception:
                pass

    def update_camera(self, camera_idx: int, cage_idx: int, has_egg: bool, animate: bool = True):
        col = self._camera_to_column.get(int(camera_idx))
        if not col:
            return
        col.update_by_scan(cage_idx, has_egg, animate=bool(animate))


class MultiCameraDetectLabel(QWidget):
    exitDetectSignal = pyqtSignal()
    startNavigationSignal = pyqtSignal()
    returnChargingSignal = pyqtSignal()

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.config = cfg
        self.frame_producer = None
        # 左侧显示模式：camera=实时画面；progress=无画面进度条（新模式）
        self.ui_left_panel_mode = "camera"
        try:
            if isinstance(self.config, dict):
                self.ui_left_panel_mode = str(self.config.get("ui_left_panel_mode", "camera")).strip().lower()
        except Exception:
            self.ui_left_panel_mode = "camera"
        # 从配置读取摄像头数量，默认4个（左右各两层）
        try:
            # 兼容最多6路（左三右三）；新场景推荐4路（左二右二）
            cfg_count = int(self.config.get('camera_count', 4)) if isinstance(self.config, dict) else 4
            self.camera_count = min(6, max(1, cfg_count))
        except Exception:
            self.camera_count = 4

        # 每路摄像头的显示名称（4路：左上/左下/右上/右下；6路：左上/左中/左下/右上/右中/右下）
        self.camera_labels = self._build_camera_labels()
        self.image_labels = []
        self.status_labels = []
        self.start_time = None
        self.task_widgets = {}  # 用于存储任务队列的控件
        self.cage_counter = 1  # 用于生成笼号和种鸭编号
        self.currently_highlighted = [None] * self.camera_count  # 跟踪每行高亮的控件
        # 新增：按摄像头行跟踪最近识别与上一次识别的笼位ID
        self.current_cage_ids = [None] * self.camera_count
        self.last_cage_ids = [None] * self.camera_count
        # 新增：四排三列的小栅格行控件缓存
        self.task_row_widgets = [{} for _ in range(self.camera_count)]

        # 添加检测结果统计
        self.total_egg_count = 0
        self.detected_cages = {}  # 存储检测到蛋的笼位信息
        # 统一的笼位图标尺寸（4路时可适当放大；6路时缩小以留出空间）
        self.cage_icon_size = QSize(80, 60) if self.camera_count <= 4 else QSize(60, 45)

        # 进度条模式参数（默认：120笼/页10/方块30px）
        self.progress_total_slots = 120
        self.progress_page_size = 10
        self.progress_square_px = 30
        self.progress_cell_gap = 40
        # 重要：客户明确要求“按扫描顺序”，因此默认使用 sequence（不从笼号解析数字）
        self.progress_index_mode = "sequence"  # auto | number | sequence
        try:
            if isinstance(self.config, dict):
                self.progress_total_slots = int(self.config.get("progress_total_slots", 120))
                self.progress_page_size = int(self.config.get("progress_page_size", 10))
                self.progress_square_px = int(self.config.get("progress_square_px", 30))
                self.progress_cell_gap = int(self.config.get("progress_cell_gap", 40))
                self.progress_index_mode = str(self.config.get("progress_index_mode", "auto")).strip().lower()
        except Exception:
            pass
        self.progress_total_slots = max(1, self.progress_total_slots)
        self.progress_page_size = max(1, min(self.progress_page_size, self.progress_total_slots))
        self.progress_square_px = max(1, self.progress_square_px)
        self.progress_cell_gap = max(0, self.progress_cell_gap)

        # 进度条状态：每路摄像头单独维护“笼号->序号”的映射与当前进度
        self._progress_id_to_idx = [dict() for _ in range(self.camera_count)]
        self._progress_next_idx = [1 for _ in range(self.camera_count)]
        self._progress_last_cage_id = [None for _ in range(self.camera_count)]
        self.progress_panel = None  # 仅在 progress 模式下创建

        #
        self.detection_started = False
        self.is_paused = True
        self._skip_return_after_stop = False
        # 停止检测时，预览图片的最大数量（避免大量结果导致UI卡死/崩溃）
        try:
            if isinstance(self.config, dict):
                self.stop_preview_max_results = int(self.config.get("stop_preview_max_results", 50))
            else:
                self.stop_preview_max_results = 50
        except Exception:
            self.stop_preview_max_results = 50

        self.initUI()
        self._setup_tooltip_style()
        try:
            # 预加载模型，避免点击开始时阻塞
            from model.MultiCameraInterface import MultiCameraInterface
            MultiCameraInterface.preload_model_async(self.config)
        except Exception:
            pass

    def _build_camera_labels(self):
        """构建每路摄像头显示名称，便于用户自定义位置标记"""
        # 默认命名随路数变化：
        # - 4路：左右各两层（左上/左下/右上/右下）
        # - 6路：左右各三层（左上/左中/左下/右上/右中/右下）
        defaults = ["左上", "左下", "右上", "右下"] if self.camera_count <= 4 else ["左上", "左中", "左下", "右上", "右中", "右下"]
        labels = []
        for i in range(self.camera_count):
            label = defaults[i] if i < len(defaults) else f"摄像头{i + 1}"
            try:
                if isinstance(self.config, dict):
                    cam_cfg = self.config.get(f'camera_{i}', {}) or {}
                    label = cam_cfg.get('display_name', cam_cfg.get('alias', label)) or label
            except Exception:
                pass
            labels.append(str(label))
        return labels

    def _resolve_camera_layout_order(self):
        """
        解析摄像头在UI中的左右/上下顺序。
        - 4路：保持历史布局（左上、左下 | 过道 | 右下、右上）
        - 6路：优先按 display_name 中的“左上/左中/左下/右上/右中/右下”排序
        """
        if self.camera_count <= 4:
            left_cams = list(range(min(2, self.camera_count)))
            if self.camera_count >= 4:
                right_cams = [3, 2]
            else:
                right_cams = list(range(2, self.camera_count))
            return left_cams, right_cams

        left_slots = [None, None, None]   # 上/中/下
        right_slots = [None, None, None]
        used = set()

        def _label_of(idx):
            try:
                if isinstance(self.camera_labels, list) and idx < len(self.camera_labels):
                    return str(self.camera_labels[idx])
            except Exception:
                pass
            return ""

        def _assign_by_keyword(idx, label):
            key_map = [
                ("左上", ("left", 0)),
                ("左中", ("left", 1)),
                ("左下", ("left", 2)),
                ("右上", ("right", 0)),
                ("右中", ("right", 1)),
                ("右下", ("right", 2)),
            ]
            for key, (side, pos) in key_map:
                if key in label:
                    if side == "left":
                        if left_slots[pos] is None:
                            left_slots[pos] = idx
                            return True
                    else:
                        if right_slots[pos] is None:
                            right_slots[pos] = idx
                            return True
            return False

        for idx in range(self.camera_count):
            label = _label_of(idx)
            if label and _assign_by_keyword(idx, label):
                used.add(idx)

        def _fill_missing(target_slots, candidates):
            for idx in candidates:
                if idx in used:
                    continue
                if None in target_slots:
                    target_slots[target_slots.index(None)] = idx
                    used.add(idx)

        # 默认顺序兜底：0-2 左侧，3-5 右侧
        _fill_missing(left_slots, list(range(min(3, self.camera_count))))
        _fill_missing(right_slots, list(range(3, self.camera_count)))

        # 仍有空位：按索引顺序补齐
        for idx in range(self.camera_count):
            if idx in used:
                continue
            if None in left_slots:
                left_slots[left_slots.index(None)] = idx
                used.add(idx)
            elif None in right_slots:
                right_slots[right_slots.index(None)] = idx
                used.add(idx)

        left_cams = [i for i in left_slots if i is not None]
        right_cams = [i for i in right_slots if i is not None]
        return left_cams, right_cams

    def _get_camera_aspect_ratio(self, camera_idx):
        """从配置读取摄像头画面宽高比（用于UI显示框比例），读取失败则回退 16:9。"""
        ratio = 16 / 9
        try:
            if isinstance(self.config, dict):
                cam_cfg = self.config.get(f'camera_{camera_idx}', {}) or {}
                w = cam_cfg.get('width')
                h = cam_cfg.get('height')
                w = float(w) if w is not None else 0.0
                h = float(h) if h is not None else 0.0
                if w > 0 and h > 0:
                    ratio = w / h
        except Exception:
            pass
        # 兜底：避免异常比例导致布局崩坏
        if ratio <= 0:
            ratio = 16 / 9
        return ratio

    def _reset_progress_state(self):
        """重置左侧进度条模式的状态（清空映射、回到第1组、全部灰色）。"""
        if not getattr(self, "progress_panel", None):
            return
        # 重置映射与游标
        self._progress_id_to_idx = [dict() for _ in range(self.camera_count)]
        self._progress_next_idx = [1 for _ in range(self.camera_count)]
        self._progress_last_cage_id = [None for _ in range(self.camera_count)]
        try:
            self.progress_panel.reset()
        except Exception:
            pass

    def _parse_cage_number(self, cage_id: str):
        """
        尝试从二维码/笼号字符串中解析出“笼位序号”：
        - cage_id 可能为 "0002-00002" / "2" / "A0002-00002" 等
        - 仅当解析结果落在 [1, progress_total_slots] 内时才认为有效
        """
        try:
            if cage_id is None:
                return None
            s = str(cage_id).strip()
            if not s:
                return None
            # 常见格式：笼号-鸭号，只取笼号部分
            if '-' in s:
                s = s.split('-', 1)[0].strip()
            m = re.search(r"(\d+)", s)
            if not m:
                return None
            n = int(m.group(1))
            if 1 <= n <= int(self.progress_total_slots):
                return n
        except Exception:
            return None
        return None

    def _infer_progress_index(self, camera_idx: int, cage_id: str):
        """将 cage_id 映射为该摄像头列的 [1..total_slots] 序号（用于点亮/滚动）。"""
        try:
            camera_idx = int(camera_idx)
        except Exception:
            return None
        if camera_idx < 0 or camera_idx >= self.camera_count:
            return None

        mode = str(self.progress_index_mode or "auto").strip().lower()
        if mode in ("number", "auto"):
            n = self._parse_cage_number(cage_id)
            if n is not None:
                return n
            if mode == "number":
                return None

        # sequence / auto fallback：按“新笼号出现的顺序”依次编号 1..N
        try:
            cid = str(cage_id).strip()
        except Exception:
            return None
        if not cid:
            return None

        mp = self._progress_id_to_idx[camera_idx]
        if cid in mp:
            return mp[cid]

        nxt = int(self._progress_next_idx[camera_idx])
        if nxt < 1 or nxt > int(self.progress_total_slots):
            return None
        mp[cid] = nxt
        self._progress_next_idx[camera_idx] = nxt + 1
        return nxt

    def _update_progress_from_detection(self, camera_idx: int, cage_id: str, egg_num: int):
        """根据检测结果驱动进度条：推进当前位置；egg_num>0 则对应方块变绿并保持。"""
        if not getattr(self, "progress_panel", None):
            return

        idx = self._infer_progress_index(camera_idx, cage_id)
        if idx is None:
            return

        has_egg = False
        try:
            has_egg = int(egg_num) > 0
        except Exception:
            has_egg = False

        # 若当前不在进度条视图（例如临时切到“调试画面”），仍然更新进度，但不做动画，避免后台白白跑动画
        in_progress_view = str(getattr(self, "ui_left_panel_mode", "camera")).lower() in ("progress", "no_camera", "no-preview", "nopreview")
        self.progress_panel.update_camera(int(camera_idx), int(idx), bool(has_egg), animate=bool(in_progress_view))

    def _setup_tooltip_style(self):
        """设置工具提示的样式"""
        # 设置全局工具提示样式
        tooltip_style = """
        QToolTip {
            background-color: #1769FF;
            color: white;
            border: none;
            padding: 10px 15px;
            font-size: 12px;
            font-weight: normal;
            border-radius: 6px;
            opacity: 0.95;
        }
        """
        QApplication.instance().setStyleSheet(QApplication.instance().styleSheet() + tooltip_style)

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 顶部标题栏
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)

        # 2. 第二行控制按钮
        control_row_widget = self._create_control_row()
        main_layout.addWidget(control_row_widget)
        main_layout.addSpacing(5)  # 减小垂直间距

        # 3. 中间内容区域
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(1)  # 减小分割条宽度
        # 保存为成员以便在窗口缩放时强制4:6比例
        self.content_splitter = content_splitter

        # 2.1 左侧摄像头画面
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)

        # 2.2 右侧实时任务队列
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)

        # 设置 splitter 比例 - 4路使用 4:6，6路适当增加左侧宽度
        left_weight = 4 if self.camera_count <= 4 else 9
        right_weight = 6 if self.camera_count <= 4 else 11
        total_weight = max(1, left_weight + right_weight)
        self._splitter_left_ratio = left_weight / total_weight
        content_splitter.setStretchFactor(0, left_weight)
        content_splitter.setStretchFactor(1, right_weight)

        main_layout.addWidget(content_splitter, 1)  # 设置拉伸因子，让其填充可用空间

        self.setLayout(main_layout)
        # 初始应用一次4:6比例
        try:
            self._apply_splitter_ratio()
        except Exception:
            pass

    def _apply_splitter_ratio(self):
        """将左右面板宽度强制为固定比例（4路4:6，6路更宽左侧）。"""
        try:
            total = max(1, self.content_splitter.width())
            left_ratio = getattr(self, "_splitter_left_ratio", 0.4)
            left = int(total * float(left_ratio))
            right = max(1, total - left)
            self.content_splitter.setSizes([left, right])
        except Exception:
            pass

    def resizeEvent(self, event):
        """窗口大小变化时保持分割比例。"""
        try:
            self._apply_splitter_ratio()
        except Exception:
            pass
        super().resizeEvent(event)

        # 应用全局样式
        self.setStyleSheet("""
            MultiCameraDetectLabel {
                background-color: #F7F7FA;
            }
            QWidget {
                color: #6F7A93;
                background-color: #F7F7FA;
            }
            QGroupBox {
                font-family: "Microsoft YaHei";
                font-size: 16px; /* 略微调小字体 */
                border: none;
                margin-top: 10px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 5px 5px;
            }
        """)

    def _create_header(self):
        """创建顶部标题栏"""
        header_widget = QWidget()
        header_widget.setFixedHeight(50)  # 调小高度
        header_widget.setStyleSheet("background-color: #FFFFFF;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 5, 16, 0)  # 减小边距

        # Logo
        logo_label = QLabel()
        pixmap = QPixmap(_res("back.png"))  # 使用 back.png 作为 logo
        logo_label.setPixmap(pixmap.scaled(120, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # 调小logo
        header_layout.addWidget(logo_label)
        header_layout.addStretch(1)

        # 标题
        title_label = QLabel("笼养种鸭产蛋记录系统")
        # 注意：Alimama ShuHeiTi Bold 字体需要系统安装
        title_label.setFont(QFont("Alimama ShuHeiTi, 700"))
        title_label.setStyleSheet(
            "font-size: 24px; color: #353F5E; font-weight: bold; background-color: transparent;")  # 调小字体
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        # 设备状态
        status_layout = QHBoxLayout()
        status_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_icon = QLabel()
        status_icon.setPixmap(
            QPixmap(_res("status.png")).scaled(18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # 状态图标
        status_icon.setStyleSheet("background-color: transparent;")
        status_label = QLabel("设备状态:")
        status_label.setStyleSheet("font-size: 12px; color: #6F7A93; background-color: transparent;")
        status_value = QLabel("就绪")
        status_value.setStyleSheet("color: #26A872; font-size: 12px; background-color: transparent;")
        status_layout.addWidget(status_icon)
        status_layout.addWidget(status_label)
        status_layout.addWidget(status_value)
        header_layout.addLayout(status_layout)

        return header_widget

    def _create_control_row(self):
        """创建第二行控制按钮"""
        control_widget = QWidget()
        control_widget.setFixedHeight(60)  # 放大高度1.5倍
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(10, 5, 10, 5)  # 调整边距
        control_widget.setStyleSheet("""
            background-color: #FFFFFF;
            border-bottom: 1px solid #E5E7EB;
            QPushButton {
                color: #5A6A8C;
                font-size: 24px; /* 放大字体1.5倍 */
                font-weight: 600;
                border: none;
                background-color: transparent;
                padding: 6px 16px; /* 放大内边距1.5倍 */
                min-height: 45px; /* 放大最小高度1.5倍 */
                border-radius: 9px; /* 放大圆角1.5倍 */
            }
            QPushButton:hover {
                color: #353F5E;
                background-color: transparent;
            }
            QPushButton:pressed {
                background-color: transparent;
                color: #353F5E;
            }
        """)

        back_button = QPushButton(" 导航界面")
        back_button.setIcon(QIcon(_res("undo.svg")))
        back_button.setIconSize(QSize(30, 30))  # 放大图标1.5倍
        back_button.setFlat(True)
        back_button.clicked.connect(self._show_navigation)

        self.download_today_button = QPushButton(" 下载当日产蛋数据")
        self.download_today_button.setIcon(QIcon(_res("downlod.png")))
        self.download_today_button.setIconSize(QSize(30, 30))
        self.download_today_button.setFlat(True)
        self.download_today_button.clicked.connect(self._on_download_today_clicked)

        self.start_pause_button = QPushButton(" 开始检测")
        self.start_pause_button.setIcon(QIcon(_res("flight.png")))
        self.start_pause_button.setIconSize(QSize(30, 30))  # 放大图标1.5倍
        self.start_pause_button.setFlat(True)
        self.start_pause_button.clicked.connect(self._on_start_pause_clicked)

        self.stop_button = QPushButton(" 结束检测")
        self.stop_button.setIcon(QIcon(_res("power.png")))
        self.stop_button.setIconSize(QSize(30, 30))  # 放大图标1.5倍
        self.stop_button.setFlat(True)
        self.stop_button.clicked.connect(self.stop_video)

        # 左侧视图切换（便于找回“摄像头画面调试界面”）
        self.left_view_toggle_button = QPushButton()
        self.left_view_toggle_button.setIcon(QIcon(_res("config.png")))  # 添加配置图标
        self.left_view_toggle_button.setFlat(True)
        self.left_view_toggle_button.setIconSize(QSize(30, 30))  # 放大图标1.5倍
        self.left_view_toggle_button.clicked.connect(self._toggle_left_panel_mode)
        self._update_left_view_toggle_button()

        control_layout.addWidget(back_button)
        # 减小拉伸因子，让右侧按钮适当左移
        control_layout.addStretch(0.5)
        control_layout.addWidget(self.download_today_button)
        control_layout.addWidget(self.start_pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.left_view_toggle_button)

        return control_widget

    def _on_download_today_clicked(self):
        """导出当日产蛋数据为Excel"""
        # 巡检未结束时提示先结束
        if getattr(self, "detection_started", False) and not getattr(self, "is_paused", True):
            QMessageBox.information(self, "提示", "请先结束检测后再下载当日产蛋数据。")
            return

        import datetime
        import os
        try:
            import pymysql
        except Exception as e:
            QMessageBox.critical(self, "错误", f"缺少pymysql，无法连接数据库: {e}")
            return
        try:
            import pandas as pd
        except Exception as e:
            QMessageBox.critical(self, "错误", f"缺少pandas，无法导出Excel: {e}")
            return

        today = datetime.date.today()
        date_str = today.strftime("%Y-%m-%d")
        file_date = today.strftime("%Y.%m.%d")
        export_dir = r"E:\产蛋数据文件夹"
        export_name = f"{file_date}产蛋数据.xlsx"
        export_path = os.path.join(export_dir, export_name)

        # 数据库连接参数（与保存逻辑一致）
        db_cfg = {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "123456",
            "db": "wenshi_eggs_record",
        }
        table_name = "duckdata1"
        map_table = "cage_duck_map"
        columns = ["id_code", "cx_wb", "cage", "centrydate", "ge", "je", "se", "be", "de", "note"]

        try:
            conn = pymysql.connect(
                host=db_cfg["host"],
                port=db_cfg["port"],
                user=db_cfg["user"],
                password=db_cfg["password"],
                db=db_cfg["db"],
                charset="utf8mb4"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据库连接失败: {e}")
            return

        try:
            select_cols = ", ".join([f"d.{c}" for c in columns])
            sql = (
                f"SELECT {select_cols} "
                f"FROM {table_name} d "
                f"LEFT JOIN {map_table} m ON d.id_code = m.id_code "
                f"WHERE d.centrydate LIKE %s "
                f"ORDER BY (m.order_index IS NULL), m.order_index ASC"
            )
            df = pd.read_sql(sql, conn, params=[f"{date_str}%"])
        except Exception as e:
            conn.close()
            QMessageBox.critical(self, "错误", f"查询当日数据失败: {e}")
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if df is None or df.empty:
            QMessageBox.information(self, "提示", f"{date_str} 未查询到产蛋数据。")
            return

        try:
            os.makedirs(export_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法创建导出目录: {e}")
            return

        # 导出Excel（不包含img字段）
        try:
            df.to_excel(export_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出Excel失败: {e}\n请确认已安装 openpyxl。")
            return

        QMessageBox.information(self, "完成", f"导出成功：{export_path}")

    def _normalize_left_panel_mode(self, mode):
        m = str(mode).strip().lower()
        return "progress" if m in ("progress", "no_camera", "no-preview", "nopreview") else "camera"

    def _is_progress_left_panel_mode(self):
        return self._normalize_left_panel_mode(getattr(self, "ui_left_panel_mode", "camera")) == "progress"

    def _update_left_view_toggle_button(self):
        """更新左侧视图切换按钮文案"""
        try:
            if not hasattr(self, "left_view_toggle_button") or self.left_view_toggle_button is None:
                return
            if self._is_progress_left_panel_mode():
                self.left_view_toggle_button.setText(" 调试画面")
                self.left_view_toggle_button.setToolTip("切换到摄像头画面（调试用）")
            else:
                self.left_view_toggle_button.setText(" 进度条")
                self.left_view_toggle_button.setToolTip("切换到笼位进度条视图")
        except Exception:
            pass

    def _set_left_panel_mode(self, mode):
        """设置左侧面板显示模式（progress / camera）。"""
        norm = self._normalize_left_panel_mode(mode)
        self.ui_left_panel_mode = norm
        try:
            if isinstance(self.config, dict):
                self.config["ui_left_panel_mode"] = norm
        except Exception:
            pass

        # 若已创建堆叠布局，则直接切换视图（不重建，保留状态）
        try:
            if hasattr(self, "left_panel_stack") and self.left_panel_stack is not None:
                self.left_panel_stack.setCurrentIndex(0 if norm == "progress" else 1)
        except Exception:
            pass

        self._update_left_view_toggle_button()
        try:
            self._apply_splitter_ratio()
        except Exception:
            pass

    def _toggle_left_panel_mode(self):
        """一键切换左侧视图：进度条 <-> 摄像头画面（调试）。"""
        self._set_left_panel_mode("camera" if self._is_progress_left_panel_mode() else "progress")

    def _create_left_panel(self):
        """创建左侧摄像头面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 0, 5, 10)  # 减小右边距

        # 用堆叠布局同时保留两种左侧视图：
        # - progress：无画面进度条（客户使用）
        # - camera：摄像头画面（调试使用）
        stack_container = QWidget()
        stack = QStackedLayout(stack_container)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        self.left_panel_stack = stack

        # 1) 进度条视图
        progress_group = QGroupBox("巡检进度")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-family: "Microsoft YaHei";
                font-size: 16px;
                font-weight: 600;
                color: #353F5E;
                border: none;
                border-left: 6px solid #1769FF;  /* 左侧高亮竖线 */
                margin-top: 0px;
                padding-top: 35px;
                padding-left: 6px;  /* 整体右移 6px */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px; /* 标题随左边线右移 */
                padding: 25px 5px 5px 5px;
                color: #353F5E;
            }
        """)

        box_layout = QVBoxLayout(progress_group)
        box_layout.setContentsMargins(8, 30, 8, 8)
        box_layout.setSpacing(10)

        left_order, right_order = self._resolve_camera_layout_order()
        self.progress_panel = CageProgressPanel(
            camera_labels=self.camera_labels,
            total_slots=self.progress_total_slots,
            page_size=self.progress_page_size,
            square_px=self.progress_square_px,
            cell_gap=self.progress_cell_gap,
            parent=progress_group,
            left_camera_order=left_order,
            right_camera_order=right_order,
        )
        box_layout.addWidget(self.progress_panel)

        # 2) 摄像头画面视图（用于调试）
        camera_group = QGroupBox("实时摄像头画面")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-family: "Microsoft YaHei";
                font-size: 16px;
                font-weight: 600;
                color: #353F5E;
                border: none;
                border-left: 6px solid #1769FF;  /* 左侧高亮竖线 */
                margin-top: 0px;
                padding-top: 35px;
                padding-left: 6px;  /* 整体右移 6px */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px; /* 标题随左边线右移 */
                padding: 25px 5px 5px 5px;
                color: #353F5E;
            }
        """)
        from PyQt5.QtWidgets import QSizePolicy
        camera_layout = QGridLayout(camera_group)
        # 4路(2x2)调试画面：
        # - 水平间距保持适中
        # - 垂直间距：按用户要求“上面两路与下面两路间隔 +100px”
        grid_gap_h = 24 if self.camera_count <= 4 else 12
        grid_gap_v = (grid_gap_h + 100) if self.camera_count <= 4 else grid_gap_h
        camera_layout.setHorizontalSpacing(grid_gap_h)
        camera_layout.setVerticalSpacing(grid_gap_v)
        camera_layout.setContentsMargins(5, 30, 5, 5)  # 减小边距

        self.image_labels = []
        # 采用2列网格布局：
        # - 4路：左上、左下 | 右下、右上
        # - 6路：左上、左中、左下 | 右上、右中、右下
        left_indices, right_indices = self._resolve_camera_layout_order()
        max_rows = max(len(left_indices), len(right_indices), 1)

        cam_to_pos = {}
        for row_idx, cam_idx in enumerate(left_indices):
            cam_to_pos[cam_idx] = (row_idx, 0)
        for row_idx, cam_idx in enumerate(right_indices):
            cam_to_pos[cam_idx] = (row_idx, 1)

        # 创建摄像头显示区域
        # - 4路：在不破坏布局前提下，把最小高度整体 +150px（用于调试更清楚）
        # - 6路：保持原尺寸（否则会挤压右侧任务队列）
        min_w = 300 if self.camera_count <= 4 else 200
        for i in range(self.camera_count):
            cam_widget = QWidget()
            cam_layout = QVBoxLayout(cam_widget)
            cam_layout.setContentsMargins(0, 0, 0, 0)
            cam_layout.setSpacing(4)

            # 摄像头标题（显示位置名称）
            title_text = self.camera_labels[i] if i < len(self.camera_labels) else f"摄像头{i + 1}"
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(
                "color: #FFFFFF; background-color: rgba(0,0,0,0.5); font-size: 12px; font-weight: 600; padding: 2px 6px; border-radius: 4px;")
            title.setFixedHeight(20)

            # 摄像头画面
            label = AspectRatioLabel(self._get_camera_aspect_ratio(i))
            label.setText(f"摄像头 {i + 1}")
            # 最小宽度用于保证文字/画面不至于过小，高度由宽高比推导
            label.setMinimumWidth(min_w)
            # 关键：保持 16:9 显示比例，避免上下“黑边”过多或画面超出布局
            try:
                label.setMinimumHeight(max(1, int(label.heightForWidth(min_w))))
            except Exception:
                pass
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #000; color: #FFF; border-radius: 6px;")
            label.setScaledContents(False)  # 保持比例
            # 不要让高度无限扩展，否则会造成 KeepAspectRatio 缩放后出现明显上下黑边
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.image_labels.append(label)

            cam_layout.addWidget(title)
            cam_layout.addWidget(label)

            row, col = cam_to_pos.get(i, (0, 0))
            camera_layout.addWidget(cam_widget, row, col)

        # 均匀分配行列：
        # - 4路时，使用底部弹性 spacer 吸收多余高度，避免画面框被纵向拉得过高
        if self.camera_count <= 4:
            camera_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding), max_rows, 0, 1, 2)
            for r in range(max_rows):
                camera_layout.setRowStretch(r, 0)
            camera_layout.setRowStretch(max_rows, 1)
        else:
            for r in range(max_rows):
                camera_layout.setRowStretch(r, 1)
        for c in range(2):
            camera_layout.setColumnStretch(c, 1)

        # 放入堆叠布局：0=progress，1=camera
        stack.addWidget(progress_group)
        stack.addWidget(camera_group)

        # 初始显示
        stack.setCurrentIndex(0 if self._is_progress_left_panel_mode() else 1)

        left_layout.addWidget(stack_container)
        # 同步按钮文案（此时堆叠已创建）
        self._update_left_view_toggle_button()
        return left_widget

    def _create_right_panel(self):
        """创建右侧实时任务队列面板"""
        # 使用外层容器+左侧色条
        right_container = QWidget()
        right_container_layout = QHBoxLayout(right_container)
        right_container_layout.setContentsMargins(0, 0, 0, 0)
        right_container_layout.setSpacing(0)

        left_blue_bar = QWidget()
        left_blue_bar.setFixedWidth(6)
        left_blue_bar.setStyleSheet("background-color: #1769FF;")

        right_widget = QWidget()
        # 关键：右侧内容需要随窗口高度自适应拉伸，否则在“全屏->点击开始检测(切换Stack视图)”时
        # 可能出现底部一行被裁切，直到下一次 resizeEvent 才恢复。
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 10, 10)

        right_container_layout.addWidget(left_blue_bar)
        right_container_layout.addWidget(right_widget)

        # 任务队列顶部控制
        task_header_widget = QWidget()
        task_header_widget.setStyleSheet("background-color: #FFFFFF;")
        task_header_layout = QHBoxLayout(task_header_widget)
        task_header_layout.setContentsMargins(10, 15, 10, 10)

        task_title_label = QLabel("实时任务队列")
        task_title_label.setStyleSheet("""
            QLabel {
                font-family: "Microsoft YaHei";
                font-size: 16px;
                font-weight: 600;
                color: #353F5E;
                padding: 0 5px 5px 5px;
            }
        """)
        task_header_layout.addWidget(task_title_label)
        task_header_layout.addStretch(1)

        range_label = QLabel("巡检范围")
        range_label.setStyleSheet("font-family: 'Microsoft YaHei'; font-size: 14px; font-weight: 500; color: #5A6A8C;")
        task_header_layout.addWidget(range_label)

        self.range_combo = QComboBox()
        self.range_combo.addItems(["A列", "B列", "C列"])
        self.range_combo.currentTextChanged.connect(self.update_task_grid)
        self.range_combo.setStyleSheet(
            "QComboBox { border: 1px solid #DCDFE6; border-radius: 4px; padding: 4px 8px; min-width: 4em; font-size: 14px; color: #353F5E; background-color: #FFFFFF; }")
        task_header_layout.addWidget(self.range_combo)
        right_layout.addWidget(task_header_widget)

        # 任务队列区域 - 新布局：[LeftGrid] [Aisle] [RightGrid]
        task_content_widget = QWidget()
        task_content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        task_content_widget.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;
                border: 1px solid #D1D5DB;
                border-radius: 4px;
            }
        """)
        task_content_layout = QVBoxLayout(task_content_widget)
        task_content_layout.setContentsMargins(4, 4, 4, 4)

        self.task_stack = QStackedLayout()

        # 1. 初始视图
        self.initial_view = QLabel()
        self.initial_view.setAlignment(Qt.AlignCenter)
        self.initial_view.setPixmap(
            QPixmap(_res("start.png")).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.initial_view.setStyleSheet("background-color: transparent; border: none;")

        # 2. 网格视图 (新设计)
        self.task_main_widget = QWidget()
        self.task_main_widget.setStyleSheet("background-color: transparent; border: none;")
        task_main_layout = QHBoxLayout(self.task_main_widget)
        task_main_layout.setContentsMargins(0, 0, 0, 0)
        task_main_layout.setSpacing(0)

        # 左侧区域 (Left Cameras)
        self.left_task_grid = QGridLayout()
        self.left_task_grid.setSpacing(8)
        left_wrapper = QWidget()
        left_wrapper.setLayout(self.left_task_grid)

        # 中间过道
        aisle_widget = QFrame()
        aisle_widget.setFixedWidth(40)
        aisle_widget.setStyleSheet(
            "border-left: 5px solid #D1D5DB; border-right: 5px solid #D1D5DB; background-color: #F0F2F5;")
        aisle_layout = QVBoxLayout(aisle_widget)
        aisle_label = QLabel("过\n道")
        aisle_label.setAlignment(Qt.AlignCenter)
        aisle_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #909399; border: none;")
        aisle_layout.addWidget(aisle_label)

        # 右侧区域 (Right Cameras)
        self.right_task_grid = QGridLayout()
        self.right_task_grid.setSpacing(8)
        right_wrapper = QWidget()
        right_wrapper.setLayout(self.right_task_grid)

        task_main_layout.addWidget(left_wrapper, 1)
        task_main_layout.addWidget(aisle_widget, 0)
        task_main_layout.addWidget(right_wrapper, 1)

        self.task_stack.addWidget(self.initial_view)
        self.task_stack.addWidget(self.task_main_widget)

        # 3. 汇总列表视图
        self.summary_widget = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_widget)
        self.summary_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_table = QTableWidget(0, 3, self.summary_widget)
        self.summary_table.setHorizontalHeaderLabels(["笼号", "鸭号", "产蛋数量"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setStyleSheet("QTableWidget { background-color: #FFFFFF; border: none; }")
        self.summary_layout.addWidget(self.summary_table)
        self.task_stack.addWidget(self.summary_widget)
        self.summary_index = 2

        # 关键：让Stack内容区垂直拉伸占满剩余高度
        task_content_layout.addLayout(self.task_stack, 1)
        right_layout.addWidget(task_content_widget, 1)

        return right_container

    def _on_start_pause_clicked(self):
        """处理开始/暂停/恢复按钮的点击事件"""
        # 如果检测从未启动过 —— 异步启动，按钮状态由 _finalize_start_video() 更新
        if not self.detection_started:
            self.start_video()
            return  # 后续状态由 _on_producer_ready -> _finalize_start_video 回调处理

        # 如果当前是暂停状态，则恢复
        if self.is_paused:
            self.resume_video()
            self.is_paused = False
            self.start_pause_button.setText(" 暂停检测")
            self.start_pause_button.setIcon(QIcon(_res("pause-one.svg")))
        # 如果当前是运行状态，则暂停
        else:
            self.pause_video()
            self.is_paused = True
            self.start_pause_button.setText(" 开始检测")
            self.start_pause_button.setIcon(QIcon(_res("flight.png")))

    def eventFilter(self, source, event):
        """事件过滤器，用于处理任务队列的悬浮提示和点击"""
        # 由于布局改变，需要调整事件过滤逻辑，或者简化直接在控件创建时绑定事件
        # 这里简化：暂不为新网格添加复杂交互，后续可按需添加
        return super().eventFilter(source, event)

    def _show_image_dialog(self, image_path, cage_id, egg_count):
        """显示放大的图片对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"检测详情 - {cage_id}")

        layout = QVBoxLayout(dialog)

        # 顶部信息栏
        info_text = f"{cage_id}  检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  检测结果: {egg_count}枚"
        info_label = QLabel(info_text)
        layout.addWidget(info_label)

        # 图片显示
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        # 调整对话框大小以适应图片
        dialog.resize(pixmap.width() + 40, pixmap.height() + 80)
        image_label.setPixmap(pixmap.scaled(dialog.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        dialog.exec_()

    def update_task_grid(self, range_text):
        """更新任务队列网格：左侧(若干列) | 过道 | 右侧(若干列)"""
        # 清空现有网格
        self._clear_layout(self.left_task_grid)
        self._clear_layout(self.right_task_grid)

        self.task_widgets = {}
        # 重置控件缓存结构：[cam_idx] -> {'current': widget, 'prev': widget, ...}
        self.task_row_widgets = [{} for _ in range(self.camera_count)]

        # 定义列映射：4路/6路按物理位置顺序映射到UI
        left_cams, right_cams = self._resolve_camera_layout_order()

        # 创建左侧网格
        self._build_side_grid(self.left_task_grid, left_cams)
        # 创建右侧网格
        self._build_side_grid(self.right_task_grid, right_cams)

        # 若处于进度条模式，切换巡检范围相当于“换一条任务”，建议同步重置左侧进度视图
        try:
            self._reset_progress_state()
        except Exception:
            pass

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def _build_side_grid(self, grid_layout, cam_indices):
        """构建单侧网格 (N列 x 3行)"""
        # 行定义：
        # Row 0: 占位/Header (灰色笼子，无号)
        # Row 1: 实时 (Current)
        # Row 2: 上一个 (Previous)

        for col_idx, cam_idx in enumerate(cam_indices):
            # 确保cam_idx在范围内
            if cam_idx >= self.camera_count:
                continue

            # Row 0: 静态灰色图标
            static_widget = self._create_task_widget("", icon_type="gray", show_text=False)
            grid_layout.addWidget(static_widget, 0, col_idx)

            # Row 1: 实时扫描 (初始灰色，无号) -> 变绿+号
            current_widget = self._create_task_widget("", icon_type="gray", show_text=True)
            grid_layout.addWidget(current_widget, 1, col_idx)
            self.task_row_widgets[cam_idx]['current'] = current_widget

            # Row 2: 上一个 (初始灰色，无号) -> 灰+号
            prev_widget = self._create_task_widget("", icon_type="gray", show_text=True)
            grid_layout.addWidget(prev_widget, 2, col_idx)
            self.task_row_widgets[cam_idx]['prev'] = prev_widget

            # 可选：添加列标题（摄像头名称）在最上方？用户未要求，但这行是"灰色笼子"。
            # 假设Row 0就是装饰。

    def _create_task_widget(self, text, icon_type="gray", show_text=True):
        """创建一个笼位控件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignCenter)

        # 文本标签
        id_label = QLabel(text)
        id_label.setAlignment(Qt.AlignCenter)
        id_label.setStyleSheet("font-size: 12px; color: #353F5E; font-weight: bold;")
        if not show_text:
            id_label.setVisible(False)  # 占位但隐藏

        # 图标标签
        icon_label = QLabel()
        pixmap_path = _res("cage_scaning.png") if icon_type == "green" else _res("cage.png")
        pixmap = QPixmap(pixmap_path)
        if not pixmap.isNull():
            icon_label.setPixmap(pixmap.scaled(self.cage_icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(id_label)
        layout.addWidget(icon_label)

        # 存储引用
        widget.setProperty("id_label", id_label)
        widget.setProperty("icon_label", icon_label)

        return widget

    def _update_row_triplet(self, camera_idx, new_cage_id):
        """更新单个摄像头的任务显示：
        Current -> 移至 Previous
        New -> 填入 Current
        """
        if camera_idx >= len(self.task_row_widgets):
            return

        widgets = self.task_row_widgets[camera_idx]
        current_w = widgets.get('current')
        prev_w = widgets.get('prev')

        if not current_w or not prev_w:
            return

        # 获取当前显示的内容
        current_id_label = current_w.property("id_label")
        current_text = current_id_label.text()

        # 仅当新ID与当前显示不同时才更新 (避免重复刷新)
        if current_text == str(new_cage_id):
            return

        # 1. 将当前 Current 移至 Previous
        if current_text:
            prev_id_label = prev_w.property("id_label")
            prev_id_label.setText(current_text)
            # Previous 始终保持灰色图标
            # (已有图标默认是gray，无需更改，除非之前改过)

        # 2. 更新 Current
        current_id_label.setText(str(new_cage_id))
        # 变绿
        current_icon = current_w.property("icon_label")
        green_pix = QPixmap(_res("cage_scaning.png"))
        if not green_pix.isNull():
            current_icon.setPixmap(green_pix.scaled(self.cage_icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 记录高亮，以便停止时恢复
        self.currently_highlighted[camera_idx] = current_w

    def _handle_detection_results(self, camera_idx, results):
        """处理检测结果"""
        print(f"收到摄像头 {camera_idx} 的检测结果: {len(results)}")

        for result in results:
            is_early = result.get('early') is True
            partial_cage_id = result.get('cage_id')
            if partial_cage_id:
                partial_cage_id = str(partial_cage_id).strip()

            if not partial_cage_id:
                continue

            # 更新UI显示
            self._update_row_triplet(camera_idx, partial_cage_id)

            # 如果有检测到蛋，更新数据
            egg_num = result.get('egg_num', 0)
            frame_path = result.get('frame_path')

            # 进度条模式：驱动左侧小方块滚动与变色（绿=有蛋且保持）
            try:
                self._update_progress_from_detection(camera_idx, partial_cage_id, egg_num)
            except Exception:
                pass

            if egg_num >= 0:  # 即使0蛋也可能需要记录
                # 找到当前显示的Current控件绑定数据
                widgets = self.task_row_widgets[camera_idx] if camera_idx < len(self.task_row_widgets) else {}
                target_widget = widgets.get('current')

                if target_widget:
                    target_widget.setProperty("egg_count", egg_num)
                    target_widget.setProperty("image_path", frame_path)
                    target_widget.setProperty("detection_camera_idx", camera_idx)

                    self.detected_cages[partial_cage_id] = {
                        'egg_num': egg_num,
                        'frame_path': frame_path,
                        'camera_idx': camera_idx,
                        'widget': target_widget
                    }

    def _handle_egg_count_update(self, total_count):
        """处理蛋数更新信号"""
        self.total_egg_count = total_count
        print(f"总蛋数更新: {total_count}")

    def update_frames(self, images):
        """更新所有摄像头的画面"""
        # 客户模式（进度条视图）下不渲染画面，节省CPU/GPU与UI刷新开销
        if self._is_progress_left_panel_mode():
            try:
                if hasattr(self, "left_panel_stack") and self.left_panel_stack is not None:
                    if self.left_panel_stack.currentIndex() == 0:
                        return
                else:
                    return
            except Exception:
                return
        for i, image in enumerate(images):
            if i < len(self.image_labels):
                try:
                    pixmap = QPixmap.fromImage(image)
                    if not pixmap.isNull():
                        # 获取label的当前尺寸
                        label_width = self.image_labels[i].width()
                        label_height = self.image_labels[i].height()

                        # 使用KeepAspectRatio保持原始比例，确保内容完整显示
                        pixmap = pixmap.scaled(
                            label_width,
                            label_height,
                            Qt.KeepAspectRatio,  # 保持宽高比，确保内容完整
                            Qt.SmoothTransformation
                        )
                        self.image_labels[i].setPixmap(pixmap)
                        self.image_labels[i].setText("")  # 清除文本
                    else:
                        pass  # 简化处理
                except Exception as e:
                    print(f"更新摄像头 {i} 画面异常: {e}")

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

            # 更新CPU使用率（非阻塞，减少UI卡顿）
            cpu_percent = psutil.cpu_percent(interval=None)
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
        """启动视频处理（异步初始化，避免 UI 卡死）"""
        try:
            if self.frame_producer is None:
                if self.config['mode'] == 1:
                    from model.utils.getUSB import get_usb_drive_paths
                    usb_paths = get_usb_drive_paths()
                    if not usb_paths:
                        QMessageBox.information(self, "提示", '检测不到U盘')
                        return
                    self.config['picture_save_path'] = usb_paths[0]

                # 禁用按钮，防止重复点击
                try:
                    self.start_pause_button.setEnabled(False)
                except Exception:
                    pass

                # 加载提示对话框（不可关闭）
                self._loading_dialog = QDialog(self)
                self._loading_dialog.setWindowTitle("正在启动")
                self._loading_dialog.setWindowFlags(
                    Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
                self._loading_dialog.setFixedSize(360, 110)
                _dlg_layout = QVBoxLayout(self._loading_dialog)
                self._loading_dot_idx = 0
                self._loading_label = QLabel("正在启动检测系统，请稍候...")
                self._loading_label.setAlignment(Qt.AlignCenter)
                self._loading_label.setStyleSheet("font-size: 15px; color: #353F5E; font-weight: bold;")
                _dlg_layout.addWidget(self._loading_label)
                _sub = QLabel("正在加载 AI 模型，约需 15~30 秒，请勿操作")
                _sub.setAlignment(Qt.AlignCenter)
                _sub.setStyleSheet("font-size: 11px; color: #888;")
                _dlg_layout.addWidget(_sub)

                self._loading_timer = QTimer(self)
                def _tick():
                    self._loading_dot_idx = (self._loading_dot_idx + 1) % 3
                    self._loading_label.setText(
                        "正在启动检测系统，请稍候" + "." * (self._loading_dot_idx + 1))
                self._loading_timer.timeout.connect(_tick)
                self._loading_timer.start(500)
                self._loading_dialog.show()
                QApplication.processEvents()

                # 后台线程构造 MultiCameraInterface，避免阻塞 UI
                self._init_thread = _ProducerInitThread(self.config, self)
                self._init_thread.succeeded.connect(self._on_producer_ready)
                self._init_thread.failed.connect(self._on_producer_failed)
                self._init_thread.start()

        except Exception as e:
            import traceback
            print(f"启动视频处理异常: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"启动失败: {e}")

    def _on_producer_ready(self, producer):
        """后台初始化成功，在 UI 线程完成信号连接和线程启动"""
        try:
            self._loading_timer.stop()
            self._loading_dialog.close()
        except Exception:
            pass
        try:
            self.start_pause_button.setEnabled(True)
        except Exception:
            pass
        try:
            self.frame_producer = producer
            self.frame_producer.frames_generated.connect(self.update_frames)
            self.frame_producer.detection_results_generated.connect(self._handle_detection_results)
            self.frame_producer.egg_count_updated.connect(self._handle_egg_count_update)

            self.task_stack.setCurrentIndex(1)
            self.update_task_grid(self.range_combo.currentText())
            try:
                QTimer.singleShot(0, self._force_relayout_after_start)
            except Exception:
                pass
            try:
                self._reset_progress_state()
            except Exception:
                pass

            self.frame_producer.start()
            self.start_time = time.time()

            # 完成启动后通知 _on_start_pause_clicked 更新按钮状态
            self._finalize_start_video()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"启动失败: {e}")

    def _on_producer_failed(self, error_msg):
        """后台初始化失败"""
        try:
            self._loading_timer.stop()
            self._loading_dialog.close()
        except Exception:
            pass
        try:
            self.start_pause_button.setEnabled(True)
        except Exception:
            pass
        self.frame_producer = None
        print(f"启动视频处理异常: {error_msg}")
        QMessageBox.critical(self, "错误", f"启动失败:\n{error_msg[:300]}")

    def _finalize_start_video(self):
        """start_video 异步完成后的收尾：更新按钮文字和状态"""
        try:
            self.detection_started = True
            self.is_paused = False
            self.start_pause_button.setText(" 暂停检测")
            self.start_pause_button.setIcon(QIcon(_res("pause-one.svg")))
        except Exception:
            pass


    def _force_relayout_after_start(self):
        """强制激活布局，修复全屏下首次切换任务队列视图的裁切问题。"""
        try:
            # 激活自身布局
            if self.layout() is not None:
                self.layout().activate()
        except Exception:
            pass
        try:
            # 激活右侧任务区布局
            if hasattr(self, "task_main_widget") and self.task_main_widget is not None:
                if self.task_main_widget.layout() is not None:
                    self.task_main_widget.layout().activate()
                self.task_main_widget.updateGeometry()
        except Exception:
            pass
        try:
            # 窗口尺寸未变化时，splitter比例不会自动重算，这里补一次
            self._apply_splitter_ratio()
        except Exception:
            pass
        try:
            self.updateGeometry()
            self.repaint()
        except Exception:
            pass

    def pause_video(self):
        if self.frame_producer is not None:
            self.frame_producer.pause()

    def resume_video(self):
        if self.frame_producer is not None:
            self.frame_producer.resume()

    def stop_video(self):
        """停止检测并显示汇总信息"""
        if self.frame_producer is not None:
            # 防止重复点击导致多次停止
            if getattr(self, "_stopping", False):
                return
            self._stopping = True
            # 停止时先断开实时信号，避免后台线程继续更新UI导致崩溃
            try:
                self.frame_producer.frames_generated.disconnect(self.update_frames)
            except Exception:
                pass
            try:
                self.frame_producer.detection_results_generated.disconnect(self._handle_detection_results)
            except Exception:
                pass
            try:
                self.frame_producer.egg_count_updated.disconnect(self._handle_egg_count_update)
            except Exception:
                pass
            # 关键修复：不要在UI线程里同步 stop()/get_detection_summary()（会 wait/join + 写盘汇总，导致卡死崩溃）
            # 改为：请求后台线程停止；后台线程完成 stop_interface + 汇总后，通过 stop_summary_ready 信号回调到UI。
            try:
                # 解除旧连接，避免多次触发
                if hasattr(self.frame_producer, "stop_summary_ready"):
                    try:
                        self.frame_producer.stop_summary_ready.disconnect(self._on_stop_summary_ready)
                    except Exception:
                        pass
                    self.frame_producer.stop_summary_ready.connect(self._on_stop_summary_ready)
            except Exception:
                pass

            try:
                self.frame_producer.stop()
            except Exception as e:
                print(f"停止检测时异常: {e}")
                # 兜底：若 stop 失败，直接用当前已有的统计显示
                summary = {'total_egg_count': self.total_egg_count, 'detected_results': self.detected_cages}
                self._finish_stop_video(summary)
                self._stopping = False
                return

            # UI提示：进入停止中状态（避免用户重复点击导致竞态）
            try:
                self.stop_button.setEnabled(False)
                self.start_pause_button.setEnabled(False)
                self.add_log("正在结束巡检并汇总结果，请稍候…")
            except Exception:
                pass
        else:
            self._finish_stop_video({'total_egg_count': self.total_egg_count, 'detected_results': self.detected_cages})

    def _on_stop_summary_ready(self, summary):
        """后台线程停止并生成汇总后回调到UI"""
        try:
            # 恢复按钮可用
            try:
                self.stop_button.setEnabled(True)
                self.start_pause_button.setEnabled(True)
            except Exception:
                pass
            self._finish_stop_video(summary if isinstance(summary, dict) else {'total_egg_count': 0, 'detected_results': {}})
        finally:
            self._stopping = False
            # 防止重复触发
            try:
                if self.frame_producer is not None and hasattr(self.frame_producer, "stop_summary_ready"):
                    self.frame_producer.stop_summary_ready.disconnect(self._on_stop_summary_ready)
            except Exception:
                pass

    def _finish_stop_video(self, summary):
        """完成停止检测的后续处理"""
        try:
            if self.frame_producer is not None:
                if self.frame_producer.isRunning():
                    self._stop_timer.start(100)
                    return

                self.frame_producer.wait(1000)
                self.frame_producer = None

            # 清空摄像头显示
            for i, label in enumerate(self.image_labels):
                label.clear()
                label.setText(f"摄像头 {i + 1}")
        except Exception as e:
            print(f"停止流程收尾异常: {e}")

        # 重置状态和按钮（即便上面异常也尽量恢复UI）
        self.detection_started = False
        self.is_paused = True
        if hasattr(self, 'start_pause_button'):
            self.start_pause_button.setText(" 开始检测")
            self.start_pause_button.setIcon(QIcon(_res("flight.png")))

        # 获取检测结果数据
        total_eggs = summary.get('total_egg_count', 0)
        detected_results = summary.get('detected_results', {})

        # 保持任务队列视图，确保任务控件可见
        if self.task_stack.currentIndex() != 1:
            self.task_stack.setCurrentIndex(1)

        # 先重置所有扫描状态的图片为灰色
        self._reset_scanning_images()

        # 不再在结束检测后加载图片到笼位图标，仅展示汇总列表

        # 最后显示检测汇总弹窗
        msg_box = QMessageBox()
        msg_box.setWindowTitle("巡检任务结束")
        msg_box.setText(f"巡检任务结束！检测蛋数：{total_eggs}枚")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        # 弹窗关闭后，切换到“汇总列表”视图并填充数据
        try:
            self._show_summary_table(detected_results)
        except Exception as e:
            print(f"显示汇总表异常: {e}")

        try:
            if not getattr(self, '_skip_return_after_stop', False):
                self.returnChargingSignal.emit()
        except Exception as e:
            print(f"结束检测后触发返回充电点失败: {e}")
        finally:
            self._skip_return_after_stop = False

        self.start_time = None

    def _show_summary_table(self, detected_results):
        """在右侧区域显示两列表格（笼号、蛋数），均匀铺满空间。"""
        try:
            if not hasattr(self, 'summary_table'):
                return
            # 填充表格
            items = list(detected_results.items()) if isinstance(detected_results, dict) else []
            # 可选排序：按笼号排序
            try:
                items.sort(key=lambda kv: str(kv[0]))
            except Exception:
                pass
            self.summary_table.clearContents()
            self.summary_table.setRowCount(len(items))
            self.summary_table.setColumnCount(3)
            self.summary_table.setHorizontalHeaderLabels(["笼号", "鸭号", "产蛋数量"])
            for row_idx, (cage_id, info) in enumerate(items):
                # 拆分笼号与鸭号（cage_id 可能为 "0001/00001" 或 "0001-00001"）
                cage_text = str(cage_id) if cage_id is not None else ""
                duck_text = ""
                if isinstance(cage_text, str):
                    sep = "/" if "/" in cage_text else ("-" if "-" in cage_text else None)
                    if sep:
                        parts = [p.strip() for p in cage_text.split(sep) if p.strip()]
                        if len(parts) >= 1:
                            cage_text = parts[0]
                        if len(parts) >= 2:
                            duck_text = parts[1]
                egg_num = 0
                try:
                    if isinstance(info, dict):
                        egg_num = int(info.get('egg_num', 0))
                    else:
                        egg_num = int(info)
                except Exception:
                    egg_num = 0

                cage_item = QTableWidgetItem(cage_text)
                duck_item = QTableWidgetItem(duck_text)
                egg_item = QTableWidgetItem(str(egg_num))
                # 居中显示
                for it in (cage_item, duck_item, egg_item):
                    it.setTextAlignment(Qt.AlignCenter)
                self.summary_table.setItem(row_idx, 0, cage_item)
                self.summary_table.setItem(row_idx, 1, duck_item)
                self.summary_table.setItem(row_idx, 2, egg_item)
            # 列宽拉伸填满
            self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            # 固定行高（不拉伸单行铺满）
            try:
                row_h = max(32, self.summary_table.fontMetrics().height() + 14)
            except Exception:
                row_h = 36
            self.summary_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            self.summary_table.verticalHeader().setDefaultSectionSize(row_h)
            # 切换到汇总视图
            if hasattr(self, 'task_stack') and hasattr(self, 'summary_index'):
                self.task_stack.setCurrentIndex(self.summary_index)
        except Exception as e:
            print(f"填充汇总表异常: {e}")

    def _reset_scanning_images(self):
        """重置所有扫描状态的图片为灰色笼位图"""
        print("开始重置所有扫描状态图片为灰色")

        # 重置所有当前高亮的控件
        for camera_idx in range(self.camera_count):
            highlighted_widget = self.currently_highlighted[camera_idx]
            if highlighted_widget:
                try:
                    image_label = highlighted_widget.property("icon_label")
                    if image_label:
                        cage_pixmap = QPixmap(_res("cage.png"))
                        if not cage_pixmap.isNull():
                            scaled_pixmap = cage_pixmap.scaled(self.cage_icon_size, Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation)
                            image_label.setPixmap(scaled_pixmap)
                            print(f"重置摄像头 {camera_idx} 的高亮控件为灰色")

                        if isinstance(image_label, DetectionImageLabel):
                            image_label.detection_info = None
                            image_label.mousePressEvent = None
                            image_label.setCursor(Qt.ArrowCursor)
                except Exception as e:
                    print(f"重置摄像头 {camera_idx} 图片时出错: {e}")

        # 清空当前高亮记录
        self.currently_highlighted = [None] * self.camera_count
        print("完成重置所有扫描状态图片")

    def _update_task_widgets_with_detection_images(self, detected_results):
        """在任务控件中显示检测图片"""
        print(f"开始更新任务控件图片，检测结果数量: {len(detected_results)}")

        for cage_id, result_info in detected_results.items():
            frame_path = result_info.get('frame_path')
            camera_idx = result_info.get('camera_idx')
            egg_num = result_info.get('egg_num', 0)
            record_time = result_info.get('record_time')

            print(f"处理笼位 {cage_id} 的检测结果，图片路径: {frame_path}")

            if frame_path and os.path.exists(frame_path) and egg_num > 0:
                target_widget = None
                if camera_idx is not None and camera_idx < len(self.task_row_widgets):
                    row_widgets = self.task_row_widgets[camera_idx]
                    target_widget = row_widgets.get('current')
                if target_widget:
                    try:
                        # 找到控件中的标签
                        id_label = target_widget.property("id_label")
                        image_label = target_widget.property("icon_label")

                        # 加载并显示检测图片
                        pixmap = QPixmap(frame_path)
                        if not pixmap.isNull():
                            scaled_pixmap = pixmap.scaled(self.cage_icon_size, Qt.KeepAspectRatio,
                                                          Qt.SmoothTransformation)

                            # 如果还不是自定义标签，则替换为自定义标签
                            if not isinstance(image_label, DetectionImageLabel):
                                new_image_label = DetectionImageLabel(target_widget)
                                new_image_label.setAlignment(Qt.AlignCenter)

                                layout = target_widget.layout()
                                if layout and image_label:
                                    layout.removeWidget(image_label)
                                    image_label.setParent(None)
                                    layout.addWidget(new_image_label)

                                image_label = new_image_label
                                target_widget.setProperty("icon_label", image_label)

                            # 设置图片
                            image_label.setPixmap(scaled_pixmap)

                            # 计算列信息和格式化显示信息
                            current_range = self.range_combo.currentText()[0] if hasattr(self, 'range_combo') else 'A'
                            column_info = f"{current_range}{camera_idx + 1}"

                            # 设置检测信息用于悬停提示
                            if isinstance(image_label, DetectionImageLabel):
                                image_label.set_detection_info(column_info, cage_id, egg_num)

                            # 存储检测信息到控件属性中，用于点击弹窗
                            target_widget.setProperty("detection_image_path", frame_path)
                            target_widget.setProperty("detection_cage_id", cage_id)
                            target_widget.setProperty("detection_egg_num", egg_num)
                            target_widget.setProperty("detection_time", record_time)
                            target_widget.setProperty("detection_camera_idx", camera_idx)

                            # 为图片标签添加点击事件
                            image_label.mousePressEvent = lambda event, widget=target_widget: self._on_detection_image_clicked(
                                widget)
                            image_label.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针为手型

                            if isinstance(id_label, QLabel):
                                id_label.setText(str(cage_id))

                            print(f"成功更新控件 cam{camera_idx}_current 的检测图片: {frame_path}")
                        else:
                            print(f"无法加载图片: {frame_path}")
                    except Exception as e:
                        print(f"更新控件图片时出错: {e}")
                else:
                    print(f"未找到笼位 {cage_id} 对应的控件")
            else:
                if not os.path.exists(frame_path) if frame_path else True:
                    print(f"笼位 {cage_id} 的图片文件不存在: {frame_path}")
                elif egg_num <= 0:
                    print(f"笼位 {cage_id} 未检测到蛋，不更新图片")

    def _on_detection_image_clicked(self, widget):
        """处理检测图片点击事件"""
        try:
            # 获取存储的检测信息
            image_path = widget.property("detection_image_path")
            cage_id = widget.property("detection_cage_id")
            egg_num = widget.property("detection_egg_num")
            record_time = widget.property("detection_time")
            camera_idx = widget.property("detection_camera_idx")

            if not image_path or not os.path.exists(image_path):
                QMessageBox.warning(self, "警告", "检测图片不存在")
                return

            # 创建弹窗显示放大图片
            self._show_detection_image_dialog(image_path, cage_id, egg_num, record_time, camera_idx)

        except Exception as e:
            print(f"处理图片点击事件时出错: {e}")
            QMessageBox.critical(self, "错误", f"无法显示图片: {str(e)}")

    def _show_detection_image_dialog(self, image_path, cage_id, egg_num, record_time, camera_idx):
        """显示检测图片弹窗"""
        dialog = QDialog(self)
        dialog.setWindowTitle("产蛋检测结果")
        dialog.setModal(True)
        dialog.resize(600, 700)

        layout = QVBoxLayout()

        # 解析笼位信息
        # cage_id格式可能是 "0181-00181" 或其他格式
        cage_parts = cage_id.split('-')
        cage_number = cage_parts[0] if cage_parts else cage_id

        # 确定列信息（基于camera_idx）
        current_range = self.range_combo.currentText()[0] if hasattr(self, 'range_combo') else 'A'
        column_info = f"{current_range}{camera_idx + 1}"

        # 格式化时间
        if record_time:
            try:
                if isinstance(record_time, (int, float)):
                    # 如果是时间戳
                    dt = datetime.datetime.fromtimestamp(record_time)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # 如果已经是字符串格式
                    formatted_time = str(record_time)
            except:
                formatted_time = "未知时间"
        else:
            formatted_time = "未知时间"

        # 创建信息栏
        info_text = f"{column_info}列  笼号：{cage_number}    检测时间：{formatted_time}   产蛋检测结果：{egg_num}枚"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 创建图片显示区域
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)

        # 加载并显示图片
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 缩放图片以适应显示区域，保持宽高比
            scaled_pixmap = pixmap.scaled(550, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
        else:
            image_label.setText("无法加载图片")

        layout.addWidget(image_label)

        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec_()

    def exit_detect(self):
        if self.frame_producer is not None:
            self.frame_producer.stop()
            self.frame_producer.wait(5000)
            self.frame_producer = None
        # self.system_monitor_timer.stop()
        self.parent().close()  # 关闭主窗口

    def _show_navigation(self):
        # 切换到导航界面（不停止检测）
        # 说明：
        # - 导航与检测通常需要并行运行；否则机器人移动时无法持续记录。
        # - 如需结束巡检，请使用“结束检测”按钮走完整停机/汇总流程。
        try:
            self.exitDetectSignal.emit()
        except Exception as e:
            print(f"发射导航切换信号失败: {e}")

    def closeEvent(self, event):
        self.exit_detect()
        event.accept()
