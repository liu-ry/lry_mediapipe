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
# 5.0 处理除了大拇指之外的其他手指
def process_finger(points, max_angle, min_angle):
    v_base = np.array(points[1]) - np.array(points[0])  # 基准向量：手腕到掌指关节
    v_mid = np.array(points[2]) - np.array(points[1])  # 中间向量：掌指关节到近端指间关节
    v_tip = np.array(points[3]) - np.array(points[0])  # 末端向量：近端指间关节到远端指间关节

    # 计算基准向量与中间向量的夹角
    cos_angle_mid = np.dot(v_base, v_mid) / (np.linalg.norm(v_base) * np.linalg.norm(v_mid))
    angle_mid_rad = np.arccos(np.clip(cos_angle_mid, -1, 1))
    angle_mid_deg = np.degrees(angle_mid_rad)

    # 计算基准向量与末端向量的夹角
    cos_angle_tip = np.dot(v_base, v_tip) / (np.linalg.norm(v_base) * np.linalg.norm(v_tip))
    angle_tip_rad = np.arccos(np.clip(cos_angle_tip, -1, 1))
    angle_tip_deg = np.degrees(angle_tip_rad)

    # 将夹角映射到0-255范围（线性映射）
    mid_val = (max_angle - angle_mid_deg) / (max_angle - min_angle) * 255
    tip_val = (max_angle - angle_tip_deg) / (max_angle - min_angle) * 255

    # 限制在0-255范围内
    mid_val = np.maximum(0, np.minimum(255, mid_val))
    tip_val = np.maximum(0, np.minimum(255, tip_val))

    return mid_val, tip_val

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

    ################ 处理食指结果映射 ################
    for hand_idx, (hand_world_landmarks, handedness) in enumerate(zip(result.hand_world_landmarks, result.handedness)):
      # 处理食指
      ff_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[5].x, hand_world_landmarks[5].y, hand_world_landmarks[5].z],  
          [hand_world_landmarks[6].x, hand_world_landmarks[6].y, hand_world_landmarks[6].z],  
          [hand_world_landmarks[9].x, hand_world_landmarks[9].y, hand_world_landmarks[9].z]]
      ff_figer_val_1, ff_figer_val_2  = process_finger(ff_finger_points, max_angle=85.9, min_angle=0)
      # 处理中指
      mf_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[9].x, hand_world_landmarks[9].y, hand_world_landmarks[9].z],  
          [hand_world_landmarks[10].x, hand_world_landmarks[10].y, hand_world_landmarks[10].z],
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z]]
      mf_figer_val_1, mf_figer_val_2  = process_finger(mf_finger_points, max_angle=85.9, min_angle=0)      
      # 处理无名指
      rf_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z],  
          [hand_world_landmarks[14].x, hand_world_landmarks[14].y, hand_world_landmarks[14].z],
          [hand_world_landmarks[17].x, hand_world_landmarks[17].y, hand_world_landmarks[17].z]] 
      rf_figer_val_1, rf_figer_val_2  = process_finger(rf_finger_points, max_angle=85.9, min_angle=0)  
      # 处理小指
      lf_finger_points = [
          [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z],
          [hand_world_landmarks[17].x, hand_world_landmarks[17].y, hand_world_landmarks[17].z],  
          [hand_world_landmarks[18].x, hand_world_landmarks[18].y, hand_world_landmarks[18].z],
          [hand_world_landmarks[13].x, hand_world_landmarks[13].y, hand_world_landmarks[13].z]] 
      lf_figer_val_1, lf_figer_val_2  = process_finger(lf_finger_points, max_angle=85.9, min_angle=0) 
    
      joint_state_msg.position = [255.0, ff_figer_val_1, mf_figer_val_1, rf_figer_val_1, lf_figer_val_1, 255.0, 10.0, 100.0, 180.0, 240.0, 245.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
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
        min_hand_detection_confidence=0.5,  # 检测置信度阈值（过滤误检）
        min_hand_presence_confidence=0.5,   # 手部存在置信度阈值
        min_tracking_confidence=0.5         # 跟踪置信度阈值（提升稳定性）
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