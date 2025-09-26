import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe import Image
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from queue import Queue
from threading import Lock

MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# maxsize=1：只保留最新一帧（避免队列堆积导致延迟）
annotated_frame_queue = Queue(maxsize=1)
# 可选：添加锁（进一步确保多线程读写安全，简单场景可省略）
queue_lock = Lock()

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def result_callback(result: vision.HandLandmarkerResult, output_image: Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    if output_image:
      print("====================================================>>>")
      annotated_image = output_image.numpy_view()  # 转为 numpy 数组（RGB 格式）
      annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
      # 3. 线程安全地存入队列（避免队列满导致阻塞）
      with queue_lock:  # 加锁确保读写安全
          if not annotated_frame_queue.full():
              annotated_frame_queue.put(result)
          else:
              # 队列满时，丢弃旧帧，存入新帧（保证显示最新画面）
              annotated_frame_queue.get()  # 移除旧帧
              annotated_frame_queue.put(result)

def main():
  base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.LIVE_STREAM,
                                        result_callback=result_callback,
                                        num_hands=2) # num_hands 用于检测手的个数

  with vision.HandLandmarker.create_from_options(options) as landmarker:
    # -------------------------- 1. 初始化 OpenCV 摄像头捕获 --------------------------
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开（避免设备占用/未连接导致报错）
    if not cap.isOpened():
      print("Error: 无法打开摄像头，请检查设备是否连接或未被占用！")
      return

    # -------------------------- 2. 循环读取摄像头帧并处理 --------------------------
    try:
      # 无限循环（直到按下 'q' 键退出）
      while True:
          # cap.read() 读取一帧图像：ret 表示读取成功与否（True/False），frame 是读取到的帧（BGR 格式，OpenCV 默认）
          ret, frame = cap.read()
          
          # 若读取失败（如摄像头断开），跳出循环
          if not ret:
              print("Error: 无法读取摄像头帧，可能设备已断开！")
              break
          
          # -------------------------- 3. 转换 OpenCV 帧为 MediaPipe 图像对象 --------------------------
          # 步骤1：OpenCV 读取的帧是 BGR 格式，而 MediaPipe 要求 SRGB 格式，需先转换通道顺序
          # cvtColor：将 BGR 转为 RGB（SRGB 与 RGB 格式在像素值范围上一致，MediaPipe 可通用）
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
          # 步骤2：将 OpenCV 的 numpy 数组（frame_rgb）转为 MediaPipe 支持的 Image 对象
          # image_format=mp.ImageFormat.SRGB：指定图像格式为 SRGB（MediaPipe 推荐格式）
          # data=frame_rgb：传入转换后的 RGB 格式 numpy 数组
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
          frame_timestamp_ms = int(time.time() * 1000)
          landmarker.detect_async(mp_image, frame_timestamp_ms)
          
          # -------------------------- 4. （可选）MediaPipe 后续处理（示例：打印图像信息） --------------------------
          # 这里可添加你的 MediaPipe 核心逻辑（如手部检测、姿态估计等）
          print(f"MediaPipe 图像尺寸：{mp_image.width}x{mp_image.height}，格式：{mp_image.image_format}")
          
          # -------------------------- 5. 显示原始摄像头帧（OpenCV 窗口） --------------------------
          # 用 OpenCV 显示原始 BGR 帧（避免再次转换通道，直接显示更高效）
          with queue_lock:
              if not annotated_frame_queue.empty():
                  # 取出标注图像
                  annotated_frame = annotated_frame_queue.get()
                  annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), annotated_frame)
                  annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                  # 显示标注结果（主线程调用，无线程冲突）
                  cv2.imshow("Hand Landmarks (Annotated)", annotated_image_bgr)
          
          # -------------------------- 6. 退出循环条件（按下 'q' 键） --------------------------
          # waitKey(1)：等待 1ms 接收键盘输入，返回值为按键的 ASCII 码（'q' 的 ASCII 码是 113）
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break  # 按下 'q' 则跳出循环
    finally:      
      # -------------------------- 7. 释放资源（避免内存泄漏） --------------------------
      cap.release()  # 释放摄像头资源
      cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

if __name__ == "__main__":
    main()
