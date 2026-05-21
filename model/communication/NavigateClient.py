# coding=utf-8
"""
    简单的导航HTTP客户端
    负责调用初始化与导航GET接口，并提供一次性状态端点探测以便后续完善“到点确认”。
"""
import json
import time
from typing import Dict, Optional, Tuple, List

import requests


class NavigateClient:
    def __init__(self, base_url: str = "http://192.168.22.126:18000", map_name: str = "wenshi_10", session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip('/')
        self.map_name = map_name
        self.session = session or requests.Session()

    def _build_url(self, path: str) -> str:
        if not path.startswith('/'):
            path = '/' + path
        return f"{self.base_url}{path}"

    def _get(self, path: str, params: Dict[str, str], timeout: int = 10) -> Tuple[int, Optional[Dict], str]:
        url = self._build_url(path)
        try:
            # Debug log
            print(f"[NavigateClient] GET {url} | Params: {params}")
            resp = self.session.get(url, params=params, timeout=timeout)
            text = resp.text or ""
            data = None
            try:
                data = resp.json()
            except Exception:
                # 不是标准JSON，返回原始文本
                pass
            return resp.status_code, data, text
        except Exception as e:
            return -1, None, f"REQUEST_ERROR: {e}"

    def _post(self, path: str, json_data: Dict, timeout: int = 10) -> Tuple[int, Optional[Dict], str]:
        url = self._build_url(path)
        try:
            # Debug log
            print(f"[NavigateClient] POST {url} | JSON: {json_data}")
            resp = self.session.post(url, json=json_data, timeout=timeout)
            text = resp.text or ""
            data = None
            try:
                data = resp.json()
            except Exception:
                # 不是标准JSON，返回原始文本
                pass
            return resp.status_code, data, text
        except Exception as e:
            return -1, None, f"REQUEST_ERROR: {e}"

    def initialize_directly_point(self, init_point_name: str = "initPoint", timeout: int = 10) -> Tuple[int, Optional[Dict], str]:
        params = {"map_name": self.map_name, "init_point_name": init_point_name}
        return self._get("/raysense-navigate/init_manager/initialize_directly_point", params, timeout=timeout)

    def navigate_point(self, position_name: str, timeout: int = 10) -> Tuple[int, Optional[Dict], str]:
        params = {"map_name": self.map_name, "position_name": position_name}
        return self._get("/raysense-navigate/nav_manager/navigate_point", params, timeout=timeout)

    def navigate_status(self, timeout: int = 5) -> Tuple[int, Optional[Dict], str]:
        return self._get("/raysense-navigate/init_manager/navigate_status", params={}, timeout=timeout)

    def navigate_set_idle(self, timeout: int = 5) -> Tuple[int, Optional[Dict], str]:
        return self._get("/raysense-navigate/nav_manager/navigate_set_idle", params={}, timeout=timeout)

    def get_robot_position(self, timeout: int = 5) -> Tuple[int, Optional[Dict], str]:
        return self._get("/raysense-navigate/init_manager/robot_position", params={}, timeout=timeout)

    def get_origin_coordinate(self, timeout: int = 5) -> Tuple[int, Optional[Dict], str]:
        params = {"map_name": self.map_name}
        return self._get("/raysense-navigate/init_manager/origin_coordinate", params=params, timeout=timeout)

    def probe_status_endpoints(self, extra_params: Optional[Dict[str, str]] = None, timeout: int = 6) -> List[Tuple[str, int, Optional[Dict], str]]:
        """
        探测常见状态/结果端点，帮助发现“到点成功”的返回信息。
        仅调用一次每个候选端点并返回原样结果，供外部记录与分析。
        """
        candidates = [
            "/raysense-navigate/nav_manager/get_status",
            "/raysense-navigate/nav_manager/nav_status",
            "/raysense-navigate/nav_manager/current_status",
            "/raysense-navigate/nav_manager/current_state",
            "/raysense-navigate/nav_manager/navigate_result",
            "/raysense-navigate/nav_manager/nav_result",
            "/raysense-navigate/nav_manager/current_task",
            "/raysense-navigate/nav_manager/current_goal",
            "/raysense-navigate/nav_manager/arrive_status",
        ]

        params = {"map_name": self.map_name}
        if extra_params:
            params.update(extra_params)

        results: List[Tuple[str, int, Optional[Dict], str]] = []
        for path in candidates:
            code, data, text = self._get(path, params=params, timeout=timeout)
            results.append((path, code, data, text))
        return results

    @staticmethod
    def is_success_response(data: Optional[Dict]) -> bool:
        if not isinstance(data, dict):
            return False
        # 常见约定：resp_code == 0 代表成功
        if str(data.get("resp_code", "")) == "0":
            return True
        # 兜底：如果有message/resp_msg包含成功关键词
        msg = str(data.get("resp_msg", "")) + str(data.get("message", ""))
        success_keywords = ["success", "successed", "succeeded", "成功", "到达"]
        return any(k.lower() in msg.lower() for k in success_keywords if msg)


