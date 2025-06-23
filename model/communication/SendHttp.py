# coding=utf-8
"""
    @project: duck_egg
    @Author：lzy
    @file： SendHttp.py
    @date：2025/6/3 13:36
"""
import requests


class SendHttp:
    def __init__(self, config):
        self.url = config["uploadUrl"]
        self.deviceId = config["deviceId"]
        self.interval = config["upload_interval"]

    def http_post(self, send_data):
        data = {'cageId': send_data['cage_id'], 'recordTime': send_data['record_time'], 'eggNum': send_data['egg_num'],
                'deviceId': self.deviceId, 'interval': self.interval}
        with open(send_data['frame_path'], 'rb') as file:
            files = {'file': file}
            response = requests.post(self.url, data=data, files=files)
            print(response)