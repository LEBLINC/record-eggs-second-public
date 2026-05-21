# coding=utf-8
"""
    多摄像头笼养蛋鸭产蛋记录系统入口
    @project: EGGRECORDQT
    @Author：lzy
    @file： main.py
"""
import sys
import traceback
import ctypes

# Windows: 若 conda 环境 site-packages 无写权限，pip 往往会把依赖装到 usersite（AppData/Roaming）。
# 为避免 usersite 中的同名包“覆盖”conda 环境内包（例如 ultralytics 版本不匹配导致 fuse/bn 异常），
# 这里将 usersite 调整到 sys.path 末尾：既能使用 usersite 中缺失的包，又能优先使用 conda 环境内包。
try:
    import site

    _usersite = site.getusersitepackages()
    if isinstance(_usersite, str) and _usersite in sys.path:
        sys.path.remove(_usersite)
        sys.path.append(_usersite)
except Exception:
    pass
import os
import torch
import shutil
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from views.Window.mainWindow import MainWindow
from model.utils.config_sample import get_default_config
from model.utils.path_utils import resource_path, resolve_resource_path
from log_utils import setup_logger
import cv2
import yaml

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


def _find_config_path() -> str:
    """Prefer editable config near CWD, fallback to bundled config."""
    cwd_cfg = os.path.join(os.getcwd(), 'configs', 'config.yaml')
    if os.path.isfile(cwd_cfg):
        return cwd_cfg
    return resource_path('configs', 'config.yaml')


def _resolve_config_paths(cfg: dict) -> dict:
    """Resolve config file paths for packaged runs."""
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


def _ensure_model_weights(cfg: dict) -> None:
    """Copy bundled model to model/weights for visibility."""
    try:
        if not isinstance(cfg, dict):
            return
        if not getattr(sys, "frozen", False):
            return
        model_path = cfg.get('modelPath')
        if not isinstance(model_path, str) or not model_path or not os.path.isfile(model_path):
            return
        weights_dir = os.path.join(os.getcwd(), 'model', 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        dst = os.path.join(weights_dir, os.path.basename(model_path))
        if not os.path.exists(dst):
            shutil.copy2(model_path, dst)
    except Exception:
        pass


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


def _setup_crash_logging():
    try:
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.getcwd()
        log_dir = os.path.join(base, "logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, "crash.log")
    except Exception:
        return None


def _to_short_path(path: str) -> str:
    try:
        if not path or not isinstance(path, str):
            return path
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
        if n and n < len(buf):
            return buf.value
    except Exception:
        pass
    return path


def main():
    """主函数"""
    # 打包环境下将工作目录固定到可执行文件所在目录，确保相对路径可用
    try:
        if getattr(sys, "frozen", False):
            os.chdir(os.path.dirname(sys.executable))
    except Exception:
        pass
    # 设置路径
    setup_paths()

    # 打印系统信息
    print_system_info()

    # 检查PyTorch环境
    check_pytorch()

    # 禁用自动缩放（必须在 QApplication 创建前设置）
    try:
        QApplication.setAttribute(Qt.AA_DisableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    except Exception:
        pass
    # 在创建 QApplication 前锁定 Qt 插件路径（避免扫描无关插件目录）
    try:
        if getattr(sys, "frozen", False):
            exe_dir = _to_short_path(os.path.dirname(sys.executable))
            qt_min = _to_short_path(os.path.join(exe_dir, "qt_plugins_min"))
            if os.path.isdir(qt_min):
                os.environ["QT_PLUGIN_PATH"] = qt_min
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _to_short_path(os.path.join(qt_min, "platforms"))
                from PyQt5.QtCore import QCoreApplication
                QCoreApplication.setLibraryPaths([qt_min])
    except Exception:
        pass

    # 创建应用
    app = QApplication(sys.argv)

    # 加载配置：优先读取 configs/config.yaml，失败再回退到默认配置
    try:
        config_file_path = _find_config_path()
        with open(config_file_path, encoding='utf-8', mode='r') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError('配置文件格式错误')
        cfg = _resolve_config_paths(cfg)
        _ensure_model_weights(cfg)
        print(f"已从配置文件加载: {config_file_path}，camera_count={cfg.get('camera_count')}")
    except Exception as e:
        print(f"加载配置文件失败，回退到默认配置：{e}")
        cfg = get_default_config()
        cfg = _resolve_config_paths(cfg)
        _ensure_model_weights(cfg)

    # 创建主窗口
    window = MainWindow(app, cfg)
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        crash_path = _setup_crash_logging()
        if crash_path:
            try:
                with open(crash_path, "a", encoding="utf-8") as f:
                    f.write("\n=== Unhandled exception ===\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
        raise