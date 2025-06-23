import os
import subprocess
import win32file
from PyQt5.QtWidgets import QMessageBox


def get_device_info(device):
    result = subprocess.run(['udevadm', 'info', '--query=property', '--name', device], stdout=subprocess.PIPE,
                            text=True)
    if result.returncode == 0:
        return result.stdout
    return ""


def is_usb_device(device):
    info = get_device_info(device)
    for line in info.split('\n'):
        if line.startswith("ID_BUS=") and "usb" in line:
            return True
    return False


def get_mounted_usb_paths():
    # 读取 /proc/mounts 文件获取当前挂载的设备
    mounted_paths = []
    media_path = "/media/pi"

    if not os.path.exists(media_path):
        return mounted_paths

    with open('/proc/mounts', 'r') as f:
        for line in f:
            parts = line.split()
            device = parts[0]
            mount_point = parts[1]

            # 检查挂载点是否在 /media/pi 下
            if mount_point.startswith(media_path):
                # 检查设备类型是否为 USB
                if is_usb_device(device):
                    mounted_paths.append(mount_point)

    return mounted_paths

def get_usb_drive_paths():
    usb_paths = []
    drive_bits = win32file.GetLogicalDrives()
    for i in range(26):  # A-Z
        mask = 1 << i
        if drive_bits & mask:
            drive_letter = f"{chr(65 + i)}:\\"
            type_ = win32file.GetDriveType(drive_letter)
            # DRIVE_REMOVABLE = 2
            if type_ == win32file.DRIVE_REMOVABLE:
                usb_paths.append(drive_letter)
    return usb_paths
