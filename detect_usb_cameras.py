#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB摄像头检测工具
用于获取当前系统中USB摄像头的详细信息，帮助配置固定的摄像头ID
"""

import cv2
import time
import subprocess
import platform
import sys

def get_system_info():
    """获取系统信息"""
    print("=" * 60)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print("=" * 60)

def detect_cameras_detailed(max_cameras=10):
    """详细检测摄像头"""
    print("\n开始详细检测摄像头...")
    print("-" * 60)
    
    available_cameras = []
    camera_details = {}
    
    for i in range(max_cameras):
        print(f"\n检测摄像头 {i}:")
        print(f"  尝试打开摄像头 {i}...")
        
        cap = None
        try:
            # 尝试不同的backend
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto")
            ]
            
            for backend, backend_name in backends:
                try:
                    print(f"    尝试 {backend_name} backend...")
                    cap = cv2.VideoCapture(i, backend)
                    
                    if cap.isOpened():
                        print(f"    ✓ 成功使用 {backend_name} 打开摄像头 {i}")
                        
                        # 获取摄像头属性
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        contrast = cap.get(cv2.CAP_PROP_CONTRAST)
                        
                        # 尝试读取一帧
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"    ✓ 成功读取帧: {frame.shape}")
                            
                            camera_details[i] = {
                                'backend': backend,
                                'backend_name': backend_name,
                                'width': width,
                                'height': height,
                                'fps': fps,
                                'brightness': brightness,
                                'contrast': contrast,
                                'frame_shape': frame.shape
                            }
                            
                            available_cameras.append(i)
                            print(f"    ✓ 摄像头 {i} 检测成功")
                            break
                        else:
                            print(f"    ✗ 无法读取帧")
                            cap.release()
                            cap = None
                    else:
                        print(f"    ✗ 无法打开摄像头")
                        if cap:
                            cap.release()
                            cap = None
                        
                except Exception as e:
                    print(f"    ✗ {backend_name} 异常: {e}")
                    if cap:
                        cap.release()
                        cap = None
                    continue
                    
        except Exception as e:
            print(f"  ✗ 摄像头 {i} 检测异常: {e}")
        finally:
            if cap:
                cap.release()
                cap = None
            
            # 短暂等待，避免设备冲突
            time.sleep(0.2)
    
    return available_cameras, camera_details

def get_usb_device_info():
    """获取USB设备信息（Windows）"""
    print("\n获取USB设备信息...")
    print("-" * 60)
    
    try:
        # 使用Windows命令获取USB设备信息
        result = subprocess.run(['wmic', 'path', 'win32_usbcontrollerdevice', 'get', 'dependent', '/format:list'], 
                              capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("USB控制器设备信息:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("无法获取USB设备信息")
            
    except Exception as e:
        print(f"获取USB设备信息失败: {e}")

def generate_config_suggestion(camera_details):
    """生成配置建议"""
    print("\n" + "=" * 60)
    print("配置建议:")
    print("=" * 60)
    
    if not camera_details:
        print("未检测到可用摄像头，请检查摄像头连接")
        return
    
    print("\n1. 在 config.yaml 中添加以下配置:")
    print("-" * 40)
    
    camera_count = len(camera_details)
    print(f"camera_count: {camera_count}")
    
    for i, details in camera_details.items():
        print(f"\ncamera_{i}:")
        print(f"  video: {i}")
        print(f"  backend: {details['backend']}  # {details['backend_name']}")
        print(f"  width: {details['width']}")
        print(f"  height: {details['height']}")
        print(f"  fps: {details['fps']:.1f}")
        print(f"  table: duckdata{i+1}")
    
    print("\n2. 修改代码中的摄像头检测逻辑:")
    print("-" * 40)
    print("在 MultiCameraInterface.py 的 CameraDetector 类中:")
    print("将 detect_available_cameras 方法修改为:")
    print("""
    def detect_available_cameras(self, max_cameras=10):
        # 直接返回固定的摄像头ID列表
        fixed_camera_ids = [0, 1, 2]  # 根据实际检测结果修改
        print(f"使用固定摄像头ID: {fixed_camera_ids}")
        return fixed_camera_ids
    """)
    
    print("\n3. 或者修改 init_cameras 方法:")
    print("-" * 40)
    print("在 MultiCameraFrameThread 类的 init_cameras 方法中:")
    print("将自动检测替换为固定ID:")
    print("""
    # 注释掉自动检测
    # self.available_cameras = self.camera_detector.detect_available_cameras()
    
    # 使用固定摄像头ID
    self.available_cameras = [0, 1, 2]  # 根据实际检测结果修改
    """)

def main():
    """主函数"""
    print("USB摄像头检测工具")
    print("用于获取摄像头信息并生成配置建议")
    
    get_system_info()
    get_usb_device_info()
    
    available_cameras, camera_details = detect_cameras_detailed()
    
    print(f"\n检测结果汇总:")
    print(f"可用摄像头数量: {len(available_cameras)}")
    print(f"摄像头ID列表: {available_cameras}")
    
    if camera_details:
        print(f"\n详细摄像头信息:")
        for cam_id, details in camera_details.items():
            print(f"摄像头 {cam_id}:")
            print(f"  Backend: {details['backend_name']} ({details['backend']})")
            print(f"  分辨率: {details['width']}x{details['height']}")
            print(f"  帧率: {details['fps']:.1f}")
            print(f"  帧形状: {details['frame_shape']}")
    
    generate_config_suggestion(camera_details)
    
    print("\n" + "=" * 60)
    print("检测完成！")
    print("请根据上述建议修改配置文件和相关代码。")
    print("=" * 60)

if __name__ == "__main__":
    main() 