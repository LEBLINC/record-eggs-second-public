# coding=utf-8
"""
    @project: duck_egg
    @Author：lzy
    @file： SendHttp.py
    @date：2025/6/3 13:36
"""
import requests
import os


class SendHttp:
    def __init__(self, config):
        self.url = config["uploadUrl"]
        self.deviceId = config["deviceId"]
        self.interval = config["upload_interval"]

    def http_post(self, send_data):
        data = {'cageId': send_data['cage_id'], 'recordTime': send_data['record_time'], 'eggNum': send_data['egg_num'],
                'deviceId': self.deviceId, 'interval': self.interval}
        
        # 检查文件是否存在
        if not os.path.exists(send_data['frame_path']):
            print(f"错误：图片文件不存在 - {send_data['frame_path']}")
            return None
            
        try:
            with open(send_data['frame_path'], 'rb') as file:
                files = {'file': file}
                
                # 添加请求头，确保正确的内容类型
                headers = {
                    'User-Agent': 'DuckEggDetectionSystem/1.0'
                }
                
                print(f"发送HTTP POST请求到: {self.url}")
                print(f"请求数据: {data}")
                print(f"文件路径: {send_data['frame_path']}")
                
                response = requests.post(self.url, data=data, files=files, headers=headers, timeout=30)
                print(f"HTTP请求响应: {response} - 状态码: {response.status_code}")
                
                # 如果是405错误，打印更详细的信息
                if response.status_code == 405:
                    print(f"HTTP 405错误 - 方法不被允许")
                    print(f"请求URL: {self.url}")
                    print(f"请求数据: {data}")
                    print(f"响应头: {response.headers}")
                    print(f"响应内容: {response.text[:500]}")  # 只打印前500字符
                    
                    # 尝试使用GET方法测试服务器是否响应
                    try:
                        test_response = requests.get(self.url, timeout=10)
                        print(f"GET请求测试响应: {test_response.status_code}")
                    except Exception as get_e:
                        print(f"GET请求测试失败: {get_e}")
                
                return response
        except requests.exceptions.Timeout:
            print(f"HTTP请求超时: {self.url}")
            return None
        except requests.exceptions.ConnectionError as ce:
            print(f"HTTP连接错误: {ce}")
            return None
        except Exception as e:
            print(f"HTTP请求异常: {e}")
            return None