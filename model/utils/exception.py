# coding=utf-8
"""
    @project: EGGRECORDQT
    @Author：wjt
    @file： exception.py
    @date：2024/12/25 14:36
"""
import functools
import traceback


def exception_handler(func):
    """装饰器：捕获异常并记录日志"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 打印异常堆栈信息
            print(f"方法 {func.__name__} 异常: {e}")
            traceback.print_exc()

    return wrapper
