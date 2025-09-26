import cv2
import numpy as np
from mediapipe import Image
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

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

# 读取原始图片（OpenCV 默认读为 BGR 格式，如需预览需保持格式一致）
img = cv2.imread("image.jpg")
if img is not None:  # 避免图片路径错误导致崩溃
    cv2.imshow("Original Image", img)  # 第一个参数是窗口名称，第二个是图像数据
    cv2.waitKey(0)  # 等待按键（0 表示无限等待，按任意键关闭窗口）
    cv2.destroyWindow("Original Image")  # 关闭当前窗口
else:
    print("Error: 无法读取图片，请检查路径是否正确！")

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1) # num_hands 用于检测手的个数
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = Image.create_from_file("image.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)
print(f"detection_result is : ", detection_result)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# 转换颜色通道：RGB → BGR（适配 OpenCV 的显示格式）
annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow("Hand Landmarks Detection Result", annotated_image_bgr)  # 显示标注结果
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口（避免残留窗口）