# coding=utf-8
"""
    @project: EGGRECORDQT
    @Author：wjt
    @file： app.py
    @date：2024/5/15 9:48
"""
import sys
from PyQt5.QtWidgets import QApplication
from views.Window.mainWindow import MainWindow
import yaml
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # 加载配置文件
    config_file_path = "configs/config.yaml"

    with open(config_file_path, encoding="utf-8", mode="r") as f:
        cfg = yaml.safe_load(f)
    app = QApplication(sys.argv)
    window = MainWindow(app, cfg)
    window.show()
    sys.exit(app.exec_())
