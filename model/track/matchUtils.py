import numpy as np
import math


def unpack_results(results):
    """
    :param results:
    :return: 去除左右边缘框
    """
    names = results[0].names
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    cls = results[0].boxes.cls.cpu().numpy().astype(int)
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy().astype(float)
    confs = np.round(confs, decimals=2)

    # 计算每个框的中心点坐标
    centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
    # 根据中心点进行排序
    sorted_indices = sorted(range(len(centers)), key=lambda i: centers[i])

    # 对所有列表按照相同的顺序重新排序
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_cls = [cls[i] for i in sorted_indices]
    sorted_track_ids = [track_ids[i] for i in sorted_indices]
    sorted_confs = [confs[i] for i in sorted_indices]

    qr_cls_indices = [i for i, c in enumerate(sorted_cls) if names[c] == 'qr']
    egg_cls_indices = [i for i, c in enumerate(sorted_cls) if names[c] == 'egg']

    qr_boxes = [sorted_boxes[i] for i in qr_cls_indices]
    qr_track_ids = [sorted_track_ids[i] for i in qr_cls_indices]

    egg_boxes = [sorted_boxes[i] for i in egg_cls_indices]
    egg_track_ids = [sorted_track_ids[i] for i in egg_cls_indices]

    egg_confs = [sorted_confs[i] for i in egg_cls_indices]
    qr_confs = [sorted_confs[i] for i in qr_cls_indices]

    return names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs


def find_index_of_id(target_id, id_array):
    indices = np.where(id_array == target_id)[0]
    if len(indices) > 0:
        return indices[0]
    else:
        return -1


def calculate_center_distance(rect1, rect2):
    # 提取矩形1的左上角和右下角坐标
    x1_1, y1_1, x2_1, y2_1 = rect1[0], rect1[1], rect1[2], rect1[3]

    # 提取矩形2的左上角和右下角坐标
    x1_2, y1_2, x2_2, y2_2 = rect2[0], rect2[1], rect2[2], rect2[3]

    # 计算矩形1的中心点坐标
    center_x1 = (x1_1 + x2_1) / 2
    center_y1 = (y1_1 + y2_1) / 2

    # 计算矩形2的中心点坐标
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2

    # 计算中心点之间的距离
    distance = ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2) ** 0.5

    return distance


def find_closest_qr_boxes_indices(target_box, boxes, cls, names, class_name="qr"):
    """
    查找给定 QR 框左右两边最接近的 QR 框索引
    :param target_box:
    :param boxes:
    :param cls:
    :param names:
    :param class_name:
    :return:
    """
    min_distance_left = float("inf")
    min_distance_right = float("inf")
    closest_box_index_left = -1
    closest_box_index_right = -1

    target_center_x = (target_box[0] + target_box[2]) / 2

    for index, (box, cl) in enumerate(zip(boxes, cls)):
        if names[cl] == class_name and box is not target_box:
            box_center_x = (box[0] + box[2]) / 2
            distance = calculate_center_distance(target_box, box)

            if box_center_x < target_center_x and distance < min_distance_left:
                min_distance_left = distance
                closest_box_index_left = index
            elif box_center_x > target_center_x and distance < min_distance_right:
                min_distance_right = distance
                closest_box_index_right = index

    return closest_box_index_left, closest_box_index_right


def calculate_max_distance(rect1, rect2):
    x1A, y1A, x2A, y2A = rect1
    x1B, y1B, x2B, y2B = rect2

    # 计算矩形框A和矩形框B之间的最远横向距离
    if x1A > x2B:
        distance_x = x1A - x2B
    elif x1B > x2A:
        distance_x = x1B - x2A
    else:
        distance_x = 0

    # 计算矩形框A和矩形框B之间的最远纵向距离
    if y1A > y2B:
        distance_y = y1A - y2B
    elif y1B > y2A:
        distance_y = y1B - y2A
    else:
        distance_y = 0

    # 计算两个矩形框的最远边距离
    distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

    return distance


def calculate_manhattan_distance(rect1, rect2):
    # 提取矩形1的左上角和右下角坐标
    x1_1, y1_1, x2_1, y2_1 = rect1[0], rect1[1], rect1[2], rect1[3]

    # 提取矩形2的左上角和右下角坐标
    x1_2, y1_2, x2_2, y2_2 = rect2[0], rect2[1], rect2[2], rect2[3]

    # 计算矩形1的中心点坐标
    center_x1 = (x1_1 + x2_1) / 2
    center_y1 = (y1_1 + y2_1) / 2

    # 计算矩形2的中心点坐标
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2

    # 计算中心点之间的哈曼顿距离
    distance = abs(center_x2 - center_x1) + abs(center_y2 - center_y1)

    return distance


def calculate_x_distance(rect1, rect2):
    # 提取矩形1的左上角和右下角坐标
    x1_1, y1_1, x2_1, y2_1 = rect1[0], rect1[1], rect1[2], rect1[3]

    # 提取矩形2的左上角和右下角坐标
    x1_2, y1_2, x2_2, y2_2 = rect2[0], rect2[1], rect2[2], rect2[3]

    # 计算矩形1的中心点坐标
    center_x1 = (x1_1 + x2_1) / 2
    center_y1 = (y1_1 + y2_1) / 2

    # 计算矩形2的中心点坐标
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2

    # 计算中心点之间的哈曼顿距离
    distance = abs(center_x2 - center_x1)

    return distance


def find_closest_rectangle(target_rect, other_rectangles):
    min_distance = float('inf')  # 初始化最小距离为无穷大
    min_index = -1  # 初始化最小距离对应的索引为-1
    for i, rect in enumerate(other_rectangles):
        distance = calculate_center_distance(target_rect, rect)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_distance, min_index


def calculate_aspect_ratio(rect):
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height
    return abs(aspect_ratio - 1)


def calculate_aspect_ratio_rotated(rect, angle):
    print(angle)
    # 解构矩形坐标
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]

    # 计算矩形的宽度和高度
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 计算旋转后的矩形的宽度和高度
    rotated_width = abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad))
    rotated_height = abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad))

    # 计算旋转后的图像比例
    aspect_ratio = rotated_width / rotated_height

    return abs(aspect_ratio - 1)


def calculate_mid(rect, image_width):
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    rect_mid = x1 + (x2 - x1) / 2
    # print(rect_mid)
    return abs(rect_mid - image_width / 2)


def adjust_bounds(bounds):
    height, width = 1080, 1920
    x1, y1, x2, y2 = bounds
    x1 -= 5
    y1 -= 5
    x2 += 5
    y2 += 5
    # 检查左边界
    if x1 < 0:
        x1 = 0
    # 检查右边界
    if x2 > width:
        x2 = width
    # 检查上边界
    if y1 < 0:
        y1 = 0
    # 检查下边界
    if y2 > height:
        y2 = height

    return x1, y1, x2, y2


def get_top_vertices(polygon):
    # 初始化上顶点列表
    top_vertices = [polygon[0], polygon[1]]

    # 遍历多边形的顶点
    for point in polygon:
        # 如果当前点的 y 坐标小于列表中最小的顶点的 y 坐标
        if point.y < top_vertices[0].y:
            top_vertices[1] = top_vertices[0]
            top_vertices[0] = point
        # 如果当前点的 y 坐标等于列表中最小的顶点的 y 坐标，但 x 坐标更小
        elif point.y == top_vertices[0].y and point.x < top_vertices[0].x:
            top_vertices[1] = top_vertices[0]
            top_vertices[0] = point
        # 如果当前点的 y 坐标等于列表中第二小的顶点的 y 坐标，但 x 坐标更小
        elif point.y == top_vertices[1].y and point.x < top_vertices[1].x:
            top_vertices[1] = point

    return top_vertices


def calculate_angle(top_left, top_right):
    # 计算两个顶点之间的水平距离和垂直距离
    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]

    # 计算旋转角度（弧度）
    angle_rad = math.atan2(dy, dx)

    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def is_rectangle_near_image_edge(rectangle, image_width, threshold):
    x1 = rectangle[0]
    x2 = rectangle[2]

    # 确保 x1 小于或等于 x2
    left, right = sorted([x1, x2])

    distance_left = left
    distance_right = image_width - right

    return min(distance_left, distance_right) <= threshold


def infer_content_based_on_neighbor(neighbor_content, position="left"):
    """
    根据相邻二维码的内容推断当前二维码的内容
    :param neighbor_content: 相邻二维码的内容
    :param position: 当前二维码相对于相邻二维码的位置，"left" 或 "right"
    :return: 推断出的当前二维码的内容
    """
    try:
        neighbor_content_number = int(neighbor_content)
        if position == "left":
            return str(neighbor_content_number + 1)
        elif position == "right":
            return str(neighbor_content_number - 1)
    except ValueError:
        # 如果相邻二维码的内容不是整数，则返回 None
        return None


def generate_random_color():
    # 生成随机的 RGB 值
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return (r, g, b)
