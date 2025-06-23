import logging
import os
import sys
from datetime import datetime


def setup_logger():
    """设置日志记录"""
    # 创建logs目录（如果不存在）
    os.makedirs('logs', exist_ok=True)

    # 设置日志文件名（使用当前日期时间）
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 添加处理器到记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger