# coding=utf-8
"""
    摄像头测试脚本
    在运行主程序前使用此脚本测试摄像头状态
    @file： camera_test.py
"""
import cv2
import time
import sys


def test_single_camera(camera_idx, backend=None):
    """测试单个摄像头"""
    print(f"\n=== 测试摄像头 {camera_idx} ===")

    backends_to_test = []
    if backend:
        backends_to_test = [backend]
    else:
        # 按优先级测试不同的backend
        backends_to_test = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends_to_test:
        backend_name = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
            cv2.CAP_ANY: "Any"
        }.get(backend, f"Backend {backend}")

        print(f"尝试 {backend_name}...")
        cap = None
        try:
            cap = cv2.VideoCapture(camera_idx, backend)
            if cap.isOpened():
                print(f"  ✓ 摄像头 {camera_idx} 打开成功")

                # 获取默认参数
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  默认参数: {width}x{height}@{fps:.1f}fps")

                # 尝试设置优化参数
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # 获取实际设置的参数
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  设置后参数: {actual_width}x{actual_height}@{actual_fps:.1f}fps")

                # 尝试读取几帧
                successful_reads = 0
                total_attempts = 10
                for i in range(total_attempts):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        successful_reads += 1
                        if i == 0:  # 只打印第一帧的详细信息
                            print(f"  第一帧: 成功, 尺寸: {frame.shape}")
                    else:
                        print(f"  第{i + 1}帧: 读取失败")
                    time.sleep(0.1)

                success_rate = (successful_reads / total_attempts) * 100
                print(f"  读取成功率: {successful_reads}/{total_attempts} ({success_rate:.1f}%)")

                if successful_reads >= total_attempts * 0.7:  # 70%成功率算通过
                    print(f"  ✓ 摄像头 {camera_idx} 测试通过! (使用 {backend_name})")
                    return True, backend
                else:
                    print(f"  ✗ 摄像头 {camera_idx} 读取成功率过低")
            else:
                print(f"  ✗ 摄像头 {camera_idx} 无法打开 ({backend_name})")

        except Exception as e:
            print(f"  ✗ 摄像头 {camera_idx} 测试异常: {e}")
        finally:
            if cap:
                cap.release()
                time.sleep(0.2)  # 让系统释放资源

    print(f"  ✗ 摄像头 {camera_idx} 所有backend测试失败")
    return False, None


def detect_all_cameras(max_cameras=10):
    """检测所有可用摄像头"""
    print(f"开始检测摄像头 (索引 0-{max_cameras - 1})...")
    print("=" * 50)

    available_cameras = []

    for i in range(max_cameras):
        success, best_backend = test_single_camera(i)
        if success:
            backend_name = {
                cv2.CAP_DSHOW: "DirectShow",
                cv2.CAP_MSMF: "Media Foundation",
                cv2.CAP_ANY: "Any"
            }.get(best_backend, f"Backend {best_backend}")

            available_cameras.append({
                'index': i,
                'backend': best_backend,
                'backend_name': backend_name
            })

    print("\n" + "=" * 50)
    print("检测结果汇总:")
    print(f"可用摄像头数量: {len(available_cameras)}")

    if available_cameras:
        print("可用摄像头列表:")
        for i, cam in enumerate(available_cameras):
            print(f"  {i + 1}. 索引 {cam['index']}: {cam['backend_name']}")
    else:
        print("❌ 未检测到任何可用摄像头!")
        print("\n可能的原因:")
        print("1. 摄像头未正确连接")
        print("2. 摄像头驱动未安装")
        print("3. 摄像头被其他程序占用")
        print("4. USB接口问题")
        print("\n建议:")
        print("- 检查摄像头连接")
        print("- 重启计算机")
        print("- 使用设备管理器检查摄像头状态")

    return available_cameras


def test_multi_camera_simultaneous(camera_indices, duration=10):
    """测试多个摄像头同时工作"""
    print(f"\n测试多摄像头同时工作 (持续{duration}秒)...")
    print("=" * 50)

    caps = []

    # 打开所有摄像头
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # 使用最稳定的backend
        if cap.isOpened():
            # 设置参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # 降低帧率减少负担
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps.append((idx, cap))
            print(f"摄像头 {idx}: 打开成功")
        else:
            print(f"摄像头 {idx}: 打开失败")

    if not caps:
        print("没有摄像头可以同时工作!")
        return False

    # 测试同时读取
    start_time = time.time()
    frame_counts = {idx: 0 for idx, _ in caps}
    error_counts = {idx: 0 for idx, _ in caps}

    print(f"\n开始测试 {len(caps)} 个摄像头同时工作...")

    while time.time() - start_time < duration:
        for idx, cap in caps:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_counts[idx] += 1
            else:
                error_counts[idx] += 1

        time.sleep(0.1)  # 100ms间隔

    # 释放资源
    for idx, cap in caps:
        cap.release()

    # 显示结果
    print(f"\n多摄像头测试结果 ({duration}秒):")
    total_success = True
    for idx, cap in caps:
        frames = frame_counts[idx]
        errors = error_counts[idx]
        total = frames + errors
        success_rate = (frames / total * 100) if total > 0 else 0

        print(f"摄像头 {idx}: {frames} 帧成功, {errors} 帧失败, 成功率: {success_rate:.1f}%")

        if success_rate < 70:  # 70%成功率为及格线
            total_success = False

    if total_success:
        print("✓ 多摄像头同时工作测试通过!")
    else:
        print("✗ 多摄像头同时工作测试失败!")
        print("建议: 减少同时使用的摄像头数量")

    return total_success


def main():
    """主函数"""
    print("摄像头系统测试工具")
    print("=" * 50)

    # 检测所有摄像头
    available_cameras = detect_all_cameras()

    if not available_cameras:
        print("\n无法继续测试，请先解决摄像头连接问题。")
        input("按回车键退出...")
        return

    # 如果有多个摄像头，测试同时工作
    if len(available_cameras) > 1:
        print(f"\n检测到 {len(available_cameras)} 个摄像头，测试是否可以同时工作...")

        # 先测试2个摄像头
        test_indices = [cam['index'] for cam in available_cameras[:2]]
        success = test_multi_camera_simultaneous(test_indices, duration=5)

        if success and len(available_cameras) > 2:
            # 如果2个成功，再测试更多
            print(f"\n2个摄像头测试成功，尝试测试所有 {len(available_cameras)} 个摄像头...")
            all_indices = [cam['index'] for cam in available_cameras]
            test_multi_camera_simultaneous(all_indices, duration=5)

    # 生成建议配置
    print("\n" + "=" * 50)
    print("建议的程序配置:")
    max_cameras = min(len(available_cameras), 4)  # 最多建议4个
    print(f"camera_count: {max_cameras}")
    print("摄像头索引映射:")
    for i in range(max_cameras):
        cam = available_cameras[i]
        print(f"  camera_{i} -> 设备索引 {cam['index']} ({cam['backend_name']})")

    print(f"\n建议先从 {min(2, len(available_cameras))} 个摄像头开始测试程序!")
    print("=" * 50)

    input("测试完成，按回车键退出...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        input("按回车键退出...")
    finally:
        # 确保释放所有资源
        cv2.destroyAllWindows()