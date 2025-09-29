# ==============================================
# 1. 导入模块：按「标准库→第三方库→ROS 2库」分类，提升可读性
# ==============================================
import time
import queue
import threading
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import Image, solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ROS 2 相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

hand_node = None# ROS 2 发布者（全局变量，供回调函数使用）
hand_publisher = None 
# ==============================================
# 2. 全局配置：集中定义常量，后续修改无需遍历代码
# ==============================================
class Config:
    # 图像标注参数
    MARGIN = 10                # 手部文字标注与边界框间距
    FONT_SIZE = 1              # 文字大小
    FONT_THICKNESS = 1         # 文字粗细
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # 左右手标注颜色（亮绿色）
    WINDOW_NAME = "Hand Landmarks (Annotated)"  # 显示窗口名称
    
    # 摄像头参数
    CAMERA_INDEX = 0           # 摄像头设备索引（默认0为内置摄像头）
    FRAME_WIDTH = 640          # 摄像头输出宽度（降低分辨率提升帧率）
    FRAME_HEIGHT = 480         # 摄像头输出高度
    
    # ROS 2 参数
    NODE_NAME = "hand_landmarker_node"  # 节点名称（规范命名，避免冲突）
    TOPIC_NAME = "cb_right_hand_control_cmd" # 话题名称（明确功能）
    QUEUE_SIZE = 10            # ROS 2 发布队列大小（平衡实时性与可靠性）
    PUB_INTERVAL = 0.1         # ROS 2 消息发布间隔（10Hz，避免高频占用资源）
    
    # 线程/队列参数
    ANNOTATE_QUEUE_MAXSIZE = 1 # 标注帧队列大小（只存最新帧，避免延迟堆积）


# ==============================================
# 3. 全局资源：线程安全队列+锁（单独定义，避免代码分散）
# ==============================================
# 存储 MediaPipe 检测结果（而非原始图像，减少内存占用）
detection_result_queue = queue.Queue(maxsize=Config.ANNOTATE_QUEUE_MAXSIZE)
# 线程锁：确保队列读写安全（多线程场景必须加锁）
queue_lock = threading.Lock()


# ==============================================
# 4. ROS 2 消息发布：独立函数+优雅退出，避免死循环
# ==============================================
def ros2_message_publisher(result_jointState: JointState):
    """
    ROS 2 消息发布函数
    :param result_jointState: 要发布的消息内容
    """
    if hand_node is None or hand_publisher is None:
      print("警告:ROS 2 节点或发布者未初始化，无法发布消息")
      return

    try:
      hand_node.get_logger().info("===============>>> 发布 ROS 2 消息中...")
      hand_publisher.publish(result_jointState)
      # hand_node.get_logger().info(f"已发布 ROS 2 消息: {result_jointState}")
    except Exception as e:
        hand_node.get_logger().error(f"发布 ROS 2 消息失败: {str(e)}")

# ==============================================
# 5. MediaPipe 结果回调：轻量处理+线程安全存队列
# ==============================================
# 处理除了大拇指之外的其他手指（根部弯曲与侧摆）
def process_base_finger(points, roll_num_1, roll_num_2):
    v_base = np.array(points[1]) - np.array(points[0])
    v_flex = np.array(points[2]) - np.array(points[1])
    v_roll = np.array(points[3]) - np.array(points[0])
    ####################################
    n = np.cross(v_base, v_roll)                            # 叉积得到法向量
    # 向量与法向量的点积 → 计算夹角
    dot_product = np.dot(v_flex, n)
    norm_v56 = np.linalg.norm(v_flex)
    norm_n = np.linalg.norm(n)
    cos_theta_with_normal = dot_product / (norm_v56 * norm_n)
    # 向量与平面的夹角 = 90° - 向量与法向量的夹角
    angle_with_plane_rad = np.pi/2 - np.arccos(np.clip(cos_theta_with_normal, -1, 1))
    angle_with_plane_deg = np.degrees(angle_with_plane_rad)
    print(f"向量5->6与A平面的夹角：{angle_with_plane_deg:.2f}°")

    # 方法：将v_56投影到A平面 → 减去法向量方向的分量
    v_56_proj = v_flex - (np.dot(v_flex, n) / np.dot(n, n)) * n
    # 计算投影后向量与v_05的点积夹角
    dot_product_proj = np.dot(v_56_proj, v_base)
    norm_v56_proj = np.linalg.norm(v_56_proj)
    norm_v05 = np.linalg.norm(v_base)
    cos_alpha = dot_product_proj / (norm_v56_proj * norm_v05)
    angle_in_plane_rad = np.arccos(np.clip(cos_alpha, -1, 1))
    angle_in_plane_deg = np.degrees(angle_in_plane_rad)
    print(f"向量5->6与向量0->5在A平面上的夹角：{angle_in_plane_deg:.2f}°")

    # 将夹角映射到0-255范围（线性映射）
    flex_val = (85.9 - angle_with_plane_deg) / 85.9 * 255
    roll_val = np.abs(angle_in_plane_deg - roll_num_1) / roll_num_2 * 255
    # 限制在0-255范围内
    flex_val = np.maximum(0, np.minimum(255, flex_val))
    roll_val = np.maximum(0, np.minimum(255, roll_val))
    ####################################

    return flex_val, roll_val
# 处理除了大拇指之外的其他手指（指尖弯曲）
def process_edge_finger(points):
    v_base = np.array(points[1]) - np.array(points[0])
    v_edge = np.array(points[3]) - np.array(points[2])

    # 计算向量夹角
    dot_product = np.dot(v_base, v_edge)
    norm_v_base = np.linalg.norm(v_base)
    norm_v_edge = np.linalg.norm(v_edge)
    cos_theta = dot_product / (norm_v_base * norm_v_edge)
    angle_rad = np.arccos(np.clip(cos_theta, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"向量2->3与向量0->1的夹角：{angle_deg:.2f}°")

    # 将夹角映射到0-255范围（线性映射）
    angle_deg = np.abs(angle_deg - 76.0) / (76.0 - 36.0) * 255.0
    # 限制在0-255范围内
    angle_deg = np.maximum(0, np.minimum(255, angle_deg))  
    print(f"映射后的值：{angle_deg:.2f}")     

    return angle_deg
# 处理大拇指
def process_thumb_finger(points):
    # 0 -> 0 索引与点对应关系
    # 1 -> 1
    # 2 -> 2
    # 3 -> 3
    # 4 -> 4
    # 5 -> 5
    # 6 -> 9
    v_01 = np.array(points[1]) - np.array(points[0])
    v_12 = np.array(points[2]) - np.array(points[1])
    v_23 = np.array(points[3]) - np.array(points[2])
    v_34 = np.array(points[4]) - np.array(points[3])
    v_05 = np.array(points[5]) - np.array(points[0])
    v_09 = np.array(points[6]) - np.array(points[0])

    # 计算平面A的法向量（通过v_05和v_09的叉积）
    normal_cross = np.cross(v_05, v_09)
    # 归一化法向量（非必需，但可减少计算误差）
    normal = normal_cross / np.linalg.norm(normal_cross) if np.linalg.norm(normal_cross) != 0 else normal_cross

    # 1、计算大拇指根部(v_01和v_12在A平面上的夹角)
    v_01_proj = v_01 - np.dot(v_01, normal) * normal  # v_01在平面A上的投影
    v_12_proj = v_12 - np.dot(v_01, normal) * normal  # v_12在平面A上的投影
    # 计算点积
    dot_product_01_02 = np.dot(v_01_proj, v_12_proj)
    # 计算模长
    norm_v01 = np.linalg.norm(v_01_proj)
    norm_v02 = np.linalg.norm(v_12_proj)
    # 计算夹角余弦值（限制在[-1, 1]范围内避免浮点误差）
    cos_theta = np.clip(dot_product_01_02 / (norm_v01 * norm_v02), -1.0, 1.0)
    thumb_base = np.degrees(np.arccos(cos_theta))
    # print(f"向量1->2与向量0->1在A平面上的夹角：{thumb_base:.2f}°")
    # 将夹角映射到0-255范围（线性映射）
    thumb_base = 255 - np.abs(thumb_base - 18.0) / (30.0 - 18.0) * 255.0   # TODO:根据实际情况调整映射范围
    # 限制在0-255范围内
    thumb_base = np.maximum(0, np.minimum(255, thumb_base))  
    # print(f"映射后的值：{thumb_base:.2f}")

    # 2、计算大拇指侧摆(v_01和v_05平面上的夹角)
    v_05_proj = v_05 - np.dot(v_01, normal) * normal  # v_05在平面A上的投影
    dot_product_01_05 = np.dot(v_01_proj, v_05_proj)
    norm_v05 = np.linalg.norm(v_05_proj)
    cos_theta_01_05 = np.clip(dot_product_01_05 / (norm_v01 * norm_v05), -1.0, 1.0)
    angle_01_05 = np.degrees(np.arccos(cos_theta_01_05))
    # print(f"向量0->5与向量0->1在A平面上的夹角：{angle_01_05:.2f}°")
    # 将夹角映射到0-255范围（线性映射）
    thumb_side = 255 - np.abs(angle_01_05 - 30.0) / (37 - 30.0) * 255.0   # TODO:根据实际情况调整映射范围（算法识别非常不准确）
    # 限制在0-255范围内
    thumb_side = np.maximum(0, np.minimum(255, thumb_side))  
    # print(f"映射后的值：{thumb_side:.2f}")

    # 3、计算大拇指横摆
    dot_product_v_n = np.dot(v_01, normal_cross)
    cos_theta_01_A = dot_product_v_n / (np.linalg.norm(v_01) * np.linalg.norm(normal_cross))
    rad = np.pi/2 - np.arccos(np.clip(cos_theta_01_A, -1.0, 1.0))
    angle_01_A = np.degrees(rad)
    # print(f"向量0->1与平面A的夹角：{angle_01_A:.2f}°")
    # 将夹角映射到0-255范围（线性映射）
    thumb_roll = 255 - np.abs(angle_01_A - 14.0) / (16.0 - 14.0) * 255.0   # TODO:根据实际情况调整映射范围
    # 限制在0-255范围内
    thumb_roll = np.maximum(0, np.minimum(255, thumb_roll))  
    # print(f"映射后的值：{thumb_roll:.2f}")    

    # 4、计算大拇指指尖
    # 计算点积
    dot_product = np.dot(v_12, v_34)
    # 计算两个向量的模长
    norm_v12 = np.linalg.norm(v_12)
    norm_v34 = np.linalg.norm(v_34)
    cos_theta = dot_product / (norm_v12 * norm_v34)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算弧度并转换为度
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    # print(f"向量2->3与向量3->4的夹角：{theta_deg:.2f}°")
    # 将夹角映射到0-255范围（线性映射）
    thumb_tip = 255 - np.abs(theta_deg - 20.0) / (85.0 - 20.0) * 255.0   # TODO:根据实际情况调整映射范围
    # 限制在0-255范围内
    thumb_tip = np.maximum(0, np.minimum(255, thumb_tip))
    # print(f"映射后的值：{thumb_tip:.2f}")

    return thumb_base, thumb_side, thumb_roll, thumb_tip


# 5.1 处理 HandLandmarkerResult 的结果，转成灵巧手接收的 JointState 消息
def result_transfer(result: vision.HandLandmarkerResult) -> JointState:
    """
    将 MediaPipe HandLandmarkerResult 转换为 ROS 2 JointState 消息
    :param result: MediaPipe 手部检测结果
    :return: JointState 消息
    """
    joint_state_msg = JointState()
    joint_state_msg.header.stamp = hand_node.get_clock().now().to_msg()
    joint_state_msg.name = [f"hand_landmark_{i}" for i in range(20)]

    for _, (hand_world_landmarks, _) in enumerate(zip(result.hand_world_landmarks, result.handedness)):
      # 处理食指 根部
      ff_base_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[5].x, hand_world_landmarks[5].y, hand_world_landmarks[5].z],  
          [hand_world_landmarks[6].x, hand_world_landmarks[6].y, hand_world_landmarks[6].z],  
          [hand_world_landmarks[9].x, hand_world_landmarks[9].y, hand_world_landmarks[9].z]]
      ff_figer_val_1 = 255.0
      ff_figer_val_2 = 10.0
      ff_figer_val_1, ff_figer_val_2  = process_base_finger(points=ff_base_finger_points, roll_num_1=10, roll_num_2=20)
      # 处理中指 根部
      mf_base_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[9].x, hand_world_landmarks[9].y, hand_world_landmarks[9].z],  
          [hand_world_landmarks[10].x, hand_world_landmarks[10].y, hand_world_landmarks[10].z],
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z]]
      mf_figer_val_1 = 255.0
      mf_figer_val_2 = 100.0
      mf_figer_val_1, mf_figer_val_2  = process_base_finger(points=mf_base_finger_points, roll_num_1 = 5, roll_num_2 = 20)      
      # 处理无名指 根部
      rf_base_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z],  
          [hand_world_landmarks[14].x, hand_world_landmarks[14].y, hand_world_landmarks[14].z],
          [hand_world_landmarks[17].x, hand_world_landmarks[17].y, hand_world_landmarks[17].z]] 
      rf_figer_val_1 = 255.0
      rf_figer_val_2 = 180.0
      rf_figer_val_1, rf_figer_val_2  = process_base_finger(points=rf_base_finger_points, roll_num_1=10, roll_num_2=5)  
      # 处理小指 根部
      lf_base_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[17].x, hand_world_landmarks[17].y, hand_world_landmarks[17].z],  
          [hand_world_landmarks[18].x, hand_world_landmarks[18].y, hand_world_landmarks[18].z],
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z]] 
      lf_figer_val_1 = 255.0
      lf_figer_val_2 = 180.0
      lf_figer_val_1, lf_figer_val_2  = process_base_finger(points=lf_base_finger_points, roll_num_1=5, roll_num_2=10) 
      # 处理食指 指尖
      ff_edge_finger_points = [
          [hand_world_landmarks[5].x, hand_world_landmarks[5].y, hand_world_landmarks[5].z],  
          [hand_world_landmarks[6].x, hand_world_landmarks[6].y, hand_world_landmarks[6].z],  
          [hand_world_landmarks[7].x, hand_world_landmarks[7].y, hand_world_landmarks[7].z],
          [hand_world_landmarks[8].x, hand_world_landmarks[8].y, hand_world_landmarks[8].z]]
      ff_edge_val = 255.0
      ff_edge_val = process_edge_finger(points=ff_edge_finger_points)
      # 处理中指 指尖
      mf_edge_finger_points = [
          [hand_world_landmarks[9].x, hand_world_landmarks[9].y, hand_world_landmarks[9].z],
          [hand_world_landmarks[10].x, hand_world_landmarks[10].y, hand_world_landmarks[10].z],
          [hand_world_landmarks[11].x, hand_world_landmarks[11].y, hand_world_landmarks[11].z],
          [hand_world_landmarks[12].x, hand_world_landmarks[12].y, hand_world_landmarks[12].z]]
      mf_edge_val = 255.0
      mf_edge_val = process_edge_finger(points=mf_edge_finger_points)
      # 处理无名指 指尖
      rf_edge_finger_points = [
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z],
          [hand_world_landmarks[14].x, hand_world_landmarks[14].y, hand_world_landmarks[14].z],
          [hand_world_landmarks[15].x, hand_world_landmarks[15].y, hand_world_landmarks[15].z],
          [hand_world_landmarks[16].x, hand_world_landmarks[16].y, hand_world_landmarks[16].z]]
      rf_edge_val = 255.0
      rf_edge_val = process_edge_finger(points=rf_edge_finger_points)
      # 处理小指 指尖
      lf_edge_finger_points = [
          [hand_world_landmarks[17].x, hand_world_landmarks[17].y, hand_world_landmarks[17].z],
          [hand_world_landmarks[18].x, hand_world_landmarks[18].y, hand_world_landmarks[18].z],
          [hand_world_landmarks[19].x, hand_world_landmarks[19].y, hand_world_landmarks[19].z],
          [hand_world_landmarks[20].x, hand_world_landmarks[20].y, hand_world_landmarks[20].z]]
      lf_edge_val = 255.0
      lf_edge_val = process_edge_finger(points=lf_edge_finger_points)
    # 处理大拇指根部、侧摆、横摆、指尖
      thumb_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[1].x, hand_world_landmarks[1].y, hand_world_landmarks[1].z],
          [hand_world_landmarks[2].x, hand_world_landmarks[2].y, hand_world_landmarks[2].z],
          [hand_world_landmarks[3].x, hand_world_landmarks[3].y, hand_world_landmarks[3].z],
          [hand_world_landmarks[4].x, hand_world_landmarks[4].y, hand_world_landmarks[4].z],
          [hand_world_landmarks[5].x, hand_world_landmarks[5].y, hand_world_landmarks[5].z],
          [hand_world_landmarks[9].x, hand_world_landmarks[9].y, hand_world_landmarks[9].z]]
      thumble_base = 255.0
      thumble_base, thumb_side, thumb_roll, thumb_tip = process_thumb_finger(thumb_finger_points)
    
    #   joint_state_msg.position = [255.0, ff_figer_val_1, mf_figer_val_1, rf_figer_val_1, lf_figer_val_1, 255.0, 127.0, 127.0, 127.0, 127.0, 245.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
      joint_state_msg.position = [255.0, ff_figer_val_1, mf_figer_val_1, rf_figer_val_1, lf_figer_val_1, 255.0, 10.0, 100.0, 180.0, 240.0, 245.0, 0.0, 0.0, 0.0, 0.0, thumb_tip, ff_edge_val, mf_edge_val, rf_edge_val, lf_edge_val]
      joint_state_msg.velocity = [0.0 for _ in range(20)]  # 可选：速度信息，暂不使用
      joint_state_msg.effort = [0.0 for _ in range(20)]

    return joint_state_msg    

# 5.2 MediaPipe 结果回调函数（LIVE_STREAM 模式必需）
def mediapipe_result_callback(
    result: vision.HandLandmarkerResult,
    output_image: Image,
    timestamp_ms: int
):
    """
    MediaPipe 手部检测结果回调函数(LIVE_STREAM 模式必需）
    :param result: 检测结果（手部关键点+左右手信息）
    :param output_image: 标注后图像（此处不用，减少内存占用）
    :param timestamp_ms: 帧时间戳（用于同步，此处简化）
    """
    # 发布识别的手势结果（可扩展为更复杂的数据结构）
    JointState_msg = result_transfer(result)
    ros2_message_publisher(JointState_msg)
    # 加锁写入队列：避免与主线程读操作冲突
    with queue_lock:
        # 队列满时丢弃旧帧，确保存最新结果（实时场景优先最新数据）
        if detection_result_queue.full():
            detection_result_queue.get()  # 移除旧帧
        detection_result_queue.put(result)  # 存入新帧


# ==============================================
# 6. 图像标注：独立函数+类型提示，提升可维护性
# ==============================================
def draw_hand_landmarks(
    rgb_image: np.ndarray,
    detection_result: vision.HandLandmarkerResult
) -> np.ndarray:
    """
    在 RGB 图像上绘制手部关键点和左右手标注
    :param rgb_image: 输入 RGB 图像(MediaPipe 输出格式）
    :param detection_result: MediaPipe 检测结果
    :return: 标注后的 RGB 图像
    """
    # 1. 复制原始图像（避免修改输入数据，防止副作用）
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape  # 图像尺寸（后续计算坐标用）
    
    # 2. 遍历每只检测到的手（支持多手检测）
    for hand_idx, (hand_landmarks, handedness) in enumerate(
        zip(detection_result.hand_landmarks, detection_result.handedness)
    ):
        # 2.1 绘制手部关键点与连接线
        # 转换关键点格式为 MediaPipe 绘图工具要求的 proto 格式
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in hand_landmarks
        ])
        
        # 调用 MediaPipe 内置绘图工具（样式统一，减少自定义代码）
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks_proto,
            connections=solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style()
        )
        
        # 2.2 绘制左右手标注文字（位置在手部最小坐标上方）
        # 计算手部关键点的最小x/y坐标（归一化坐标→像素坐标）
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width) - Config.MARGIN
        text_y = int(min(y_coords) * height) - Config.MARGIN
        
        # 确保文字不超出图像边界（避免OpenCV报错）
        text_x = max(Config.MARGIN, text_x)
        text_y = max(Config.MARGIN, text_y)
        
        # 绘制文字（用 OpenCV 绘图，兼容显示流程）
        cv2.putText(
            img=annotated_image,
            text=f"Hand {hand_idx+1}: {handedness[0].category_name}",  # 标注手的序号+左右手
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=Config.FONT_SIZE,
            color=Config.HANDEDNESS_TEXT_COLOR,
            thickness=Config.FONT_THICKNESS,
            lineType=cv2.LINE_AA  # 抗锯齿，文字更清晰
        )
    
    return annotated_image


# ==============================================
# 7. 主函数：结构化流程+完整错误处理+资源释放
# ==============================================
def main():
    # -------------------------- 7.1 初始化 ROS 2 节点与发布者
    rclpy.init()  # 初始化 rclpy（必须在创建节点前执行）
    global hand_node, hand_publisher  # 声明使用全局变量
    hand_node = Node(Config.NODE_NAME)  # 创建 ROS 2 节点
    hand_publisher = hand_node.create_publisher(
        msg_type=JointState,
        topic=Config.TOPIC_NAME,
        qos_profile=Config.QUEUE_SIZE  # QoS 配置：平衡实时性与可靠性
    )
    hand_node.get_logger().info(f"ROS 2 节点 {Config.NODE_NAME} 已初始化")

    # -------------------------- 7.2 初始化 MediaPipe 手部检测器
    # 1. 配置检测器参数（BaseOptions 指向模型文件）
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    detector_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,  # 实时流模式（适合摄像头）
        result_callback=mediapipe_result_callback,   # 结果回调函数
        num_hands=2,  # 最多检测2只手（符合常规场景）
        min_hand_detection_confidence=0.95,  # 检测置信度阈值（过滤误检）
        min_hand_presence_confidence=0.95,   # 手部存在置信度阈值
        min_tracking_confidence=0.95         # 跟踪置信度阈值（提升稳定性）
    )

    # 2. 创建检测器（with 语句：自动释放资源，避免泄漏）
    with vision.HandLandmarker.create_from_options(detector_options) as hand_detector:
        hand_node.get_logger().info("MediaPipe 手部检测器已初始化")

        # -------------------------- 7.3 初始化 OpenCV 摄像头
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        # 设置摄像头分辨率（降低分辨率提升帧率，减少CPU占用）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        # 检查摄像头是否成功打开（关键错误处理，避免后续崩溃）
        if not cap.isOpened():
            hand_node.get_logger().fatal(f"无法打开摄像头（索引：{Config.CAMERA_INDEX}），请检查设备！")
            # 清理资源后退出
            cap.release()
            hand_node.destroy_node()
            rclpy.shutdown()
            return
        hand_node.get_logger().info(f"摄像头已打开（分辨率：{Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}）")

        # -------------------------- 7.4 主循环：摄像头读取→检测→显示
        try:
            while True:
                # -------------------------- 7.4.1 读取摄像头帧
                # ret：是否读取成功；frame：BGR格式帧（OpenCV默认）
                ret, frame_bgr = cap.read()
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"=======>>>>>>> 摄像头帧率：{fps:.2f} FPS") 
                if not ret:
                    hand_node.get_logger().warn("无法读取摄像头帧，重试...")
                    time.sleep(0.1)  # 重试前休眠，避免高频报错
                    continue

                # -------------------------- 关键添加：非阻塞式自旋，启动ROS 2事件循环
                # timeout_sec=0.001：最多等待1ms，不阻塞摄像头读取（可根据帧率调整）
                rclpy.spin_once(hand_node, timeout_sec=0.001)
                
                # -------------------------- 7.4.2 帧格式转换（OpenCV→MediaPipe）
                # MediaPipe 要求 SRGB 格式，需将 BGR→RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # 转换为 MediaPipe Image 对象（指定格式，避免歧义）
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )

                # -------------------------- 7.4.3 异步检测（不阻塞主线程）
                # 时间戳：用当前时间（毫秒级），确保检测时序正确
                frame_timestamp = int(time.time() * 1000)
                hand_detector.detect_async(mp_image, frame_timestamp)

                # -------------------------- 7.4.4 读取检测结果并标注
                with queue_lock:
                    # 队列非空时才处理（避免阻塞）
                    if not detection_result_queue.empty():
                        detection_result = detection_result_queue.get()
                        # 绘制标注（用 RGB 帧，避免重复转换）
                        annotated_frame_rgb = draw_hand_landmarks(frame_rgb, detection_result)
                        # 转换为 BGR 格式（OpenCV 显示要求）
                        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
                    else:
                        # 无检测结果时显示原始帧（避免窗口黑屏）
                        annotated_frame_bgr = frame_bgr

                # -------------------------- 7.4.5 显示图像（主线程执行，避免窗口崩溃）
                cv2.imshow(Config.WINDOW_NAME, annotated_frame_bgr)

                # -------------------------- 7.4.6 退出条件（按下 'q' 键）
                # waitKey(1)：等待1ms，确保窗口响应；0xFF：兼容不同平台
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    hand_node.get_logger().info("用户按下 'q' 键，准备退出...")
                    break

        # -------------------------- 7.5 异常处理（覆盖常见场景）
        except KeyboardInterrupt:
            hand_node.get_logger().info("捕获键盘中断（Ctrl+C）")
        except Exception as e:
            # 捕获所有未知异常，避免程序崩溃无提示
            hand_node.get_logger().error(f"主循环异常: {str(e)}", throttle_duration_sec=1.0)
        finally:
            # -------------------------- 7.6 资源释放（必须执行，避免内存泄漏）
            # 1. 释放摄像头
            cap.release()
            # 2. 关闭所有 OpenCV 窗口
            cv2.destroyAllWindows()
            # 3. 销毁 ROS 2 节点
            hand_node.destroy_node()
            # 4. 关闭 rclpy
            rclpy.shutdown()
            hand_node.get_logger().info("所有资源已释放，程序退出")


# ==============================================
# 8. 程序入口：确保主线程执行
# ==============================================
if __name__ == "__main__":
    main()