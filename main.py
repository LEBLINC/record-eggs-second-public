# coding=utf-8
"""
    多摄像头笼养蛋鸭产蛋记录系统入口
    @project: EGGRECORDQT
    @Author：lzy
    @file： main.py
"""
import sys
import os
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from views.Window.mainWindow import MainWindow
from model.utils.config_sample import get_default_config
from log_utils import setup_logger
import cv2

# 全局抑制OpenCV错误日志 - 加速初始化（兼容不同OpenCV版本）
try:
    # 尝试新版本的日志级别设置
    if hasattr(cv2, 'LOG_LEVEL_ERROR'):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    elif hasattr(cv2, 'logging') and hasattr(cv2.logging, 'LOG_LEVEL_ERROR'):
        cv2.setLogLevel(cv2.logging.LOG_LEVEL_ERROR)
    else:
        # 使用数值设置（ERROR级别通常是3）
        cv2.setLogLevel(3)
except Exception as e:
    print(f"警告：无法设置OpenCV日志级别: {e}")

# 减少DirectShow相关警告
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_DSHOW', '1')

# 设置日志
logger = setup_logger()


def setup_paths():
    """设置必要的路径"""
    # 确保必要的目录存在
    required_dirs = [
        'logs',
        'data',
        'model/weights',
    ]

    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # 添加当前目录到搜索路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)


def check_pytorch():
    """检查PyTorch环境"""
    # 打印PyTorch版本
    print(f"PyTorch version: {torch.__version__}")

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA available: 是，设备数量: {device_count}")

        # 打印每个设备的信息
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"  设备 {i}: {device_name}, CUDA能力: {device_capability}")

        # 打印第一个设备的内存信息
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # 转为GB
        mem_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"  GPU内存: 总计 {mem_total:.2f}GB, 已分配 {mem_allocated:.2f}GB, 已保留 {mem_reserved:.2f}GB")
    else:
        print("CUDA available: 否，将使用CPU")

    # 如果有CUDA，设置一些优化选项
    if torch.cuda.is_available():
        # 启用cuDNN自动调谐，提高性能
        torch.backends.cudnn.benchmark = True


def initialize_gpu_optimization():
    """初始化GPU优化"""
    if torch.cuda.is_available():
        # 设置GPU优化参数
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 清空GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 创建GPU管理器
        from model.utils.gpu_manager import GPUManager
        gpu_manager = GPUManager()

        # 打印GPU信息
        mem_info = gpu_manager.get_memory_info()
        print(f"GPU内存: 总计 {mem_info['total']:.2f}GB, "
              f"已用 {mem_info['used']:.2f}GB, "
              f"空闲 {mem_info['free']:.2f}GB")

        return gpu_manager
    else:
        print("警告: GPU不可用，将使用CPU模式")
        return None


def print_system_info():
    """打印系统信息"""
    import platform

    print("=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"处理器: {platform.processor()}")

    # 尝试获取更多Windows特定信息
    if platform.system() == 'Windows':
        try:
            import wmi
            c = wmi.WMI()
            for os_info in c.Win32_OperatingSystem():
                print(f"Windows版本: {os_info.Caption}")
                print(f"可用物理内存: {float(os_info.FreePhysicalMemory) / 1024:.2f} GB")
                print(f"总物理内存: {float(os_info.TotalVisibleMemorySize) / 1024:.2f} GB")
        except ImportError:
            print("无法获取详细的Windows系统信息（缺少wmi模块）")


    print("=================")


def main():
    """主函数"""
    # 设置路径
    setup_paths()

    # 打印系统信息
    print_system_info()

    # 检查PyTorch环境
    check_pytorch()

    # 创建应用
    app = QApplication(sys.argv)

    # 禁用自动缩放
    app.setAttribute(Qt.AA_DisableHighDpiScaling)

    # 加载配置
    cfg = get_default_config()

    # 创建主窗口
    window = MainWindow(app, cfg)
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()