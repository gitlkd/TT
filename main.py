# 导入一些必要的库
import cv2
import numpy as np
import threading
import queue
from queue import Queue
import torch
import onnxruntime as ort
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import face_alignment

# 创建一个队列，用来传递图像数据
q = Queue()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 加载yolov5模型，并转换为onnx格式，这样可以加速模型的加载和预测
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\桌面D盘\Yolov5-deepsort-driverDistracted-driving-behavior-detection-1.0\weights\\best.pt')
results = model.detect(torch.zeros((1, 3, 640, 640))) # 运行一次模型，生成结果
results.save(‘yolov5s.onnx’) # 保存结果为onnx文件
model.eval()  # 设置模型为评估模式
model(torch.zeros((1, 3, 640, 640)))  # 运行一次模型，生成onnx文件
session = ort.InferenceSession('yolov5s.onnx')  # 创建一个onnx会话

# 加载face_alignment模型，用来检测人脸和关键点，这个模型比dlib更快更准确
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False)

# 定义一个函数来计算EAR值
def get_ear(eye_points, facial_landmarks):
    # 计算眼睛的高度和宽度
    left_point = facial_landmarks[eye_points[0]]
    right_point = facial_landmarks[eye_points[3]]
    center_top = facial_landmarks[eye_points[1]]
    center_bottom = facial_landmarks[eye_points[5]]

    eye_width = np.linalg.norm(np.array(left_point) - np.array(right_point))
    eye_height = np.linalg.norm(np.array(center_top) - np.array(center_bottom))

    # 计算EAR值
    ear = eye_height / eye_width

    return ear

# 定义一个函数来计算MAR值
def get_mar(mouth_points, facial_landmarks):
    # 计算嘴巴的宽度和高度
    left_point = facial_landmarks[mouth_points[0]]
    right_point = facial_landmarks[mouth_points[1]]
    center_top = facial_landmarks[mouth_points[2]]
    center_bottom = facial_landmarks[mouth_points[3]]

    mouth_width = np.linalg.norm(np.array(left_point) - np.array(right_point))
    mouth_height = np.linalg.norm(np.array(center_top) - np.array(center_bottom))

    # 计算MAR值
    mar = mouth_height / mouth_width

    return mar


# 定义左右眼的关键点索引
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

# 定义嘴巴的关键点索引
mouth_landmarks = [49, 55, 52, 58]

# 定义一个滑动窗口来统计perclos值
window_size = 10  # 可以根据需要调整窗口大小
window = []  # 存储窗口内的眼睛状态，1表示闭合，0表示开启

# 定义一个阈值来判断眼睛是否闭合
eye_close_threshold = 0.2  # 可以根据需要调整阈值

# 定义一个阈值来判断是否为疲劳驾驶
perclos_threshold = 0.5  # 可以根据需要调整阈值

# 定义一个阈值来判断嘴巴是否张开
mouth_open_threshold = 1.02  # 可以根据需要调整阈值

# 定义一个变量来存储上一帧的嘴巴状态，0表示闭合，1表示张开
last_mouth_state = [0] * 30 # 把这个列表放在循环外面初始化，可以根据需要调整列表的长度

def program1():
    while True:
        # 捕获图像
        ret, frame = cap.read()
        # 检查捕获是否成功
        if not ret:
            print("Failed to capture frame")
            continue
        _, frame1 = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 对图像进行高斯滤波和直方图均衡化
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)

        # 使用face_alignment模型来检测人脸和关键点
        landmarks = fa.get_landmarks(gray)

        if landmarks is not None:
            # 获取第一个人脸的关键点
            landmarks = landmarks[0]

            # 计算左右眼的EAR值
            left_eye_ear = get_ear(left_eye_landmarks, landmarks)
            right_eye_ear = get_ear(right_eye_landmarks, landmarks)

            # 计算平均的EAR值
            ear = (left_eye_ear + right_eye_ear) / 2

            # 判断眼睛是否闭合，并更新窗口内的状态
            if ear < eye_close_threshold:
                eye_state = 1  # 眼睛闭合
            else:
                eye_state = 0  # 眼睛开启

            window.append(eye_state)

            # 如果窗口满了，就计算perclos值，并判断是否为疲劳驾驶
            if len(window) == window_size:
                perclos = sum(window) / window_size  # perclos为窗口内眼睛闭合的时间比例
                if perclos > perclos_threshold:
                    fatigue_state = "疲劳驾驶"  # 疲劳驾驶
                else:
                    fatigue_state = "正常驾驶"  # 正常驾驶

                # 在图像上显示perclos值和疲劳状态
                cv2.putText(frame1, "PERCLOS: {:.2f}".format(perclos), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                font_path = "simhei.ttf"  # 这里你可以用其他的中文字体文件
                font = ImageFont.truetype(font_path, 32)  # 这里你可以调整字体大小
                pil_img = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))  # 转换颜色空间
                draw = ImageDraw.Draw(pil_img)  # 创建一个画笔对象
                draw.text((20, 80), "驾驶状态: {}".format(fatigue_state), font=font, fill=(0, 255, 0))  # 在图像上写中文
                frame1 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 把PIL的图像转换回OpenCV的图像

                # 如果检测到疲劳驾驶，就发出警报声音
                if fatigue_state == "疲劳驾驶":
                    font_path = "simhei.ttf"  # 这里你可以用其他的中文字体文件
                    font = ImageFont.truetype(font_path, 55)  # 这里你可以调整字体大小
                    pil_img = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))  # 转换颜色空间
                    draw = ImageDraw.Draw(pil_img)  # 创建一个画笔对象
                    draw.text((200, 200), "快醒醒！", font=font, fill=(255, 0, 0))  # 在图像上写中文
                    frame1 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 把PIL的图像转换回OpenCV的图像
                    # 这里可以添加一个播放声音文件的函数，比如使用playsound模块

                # 移除窗口内的第一个状态，为下一帧做准备
                window.pop(0)

            # 在每一帧上画出人脸和眼睛和嘴巴的关键点
            for n in range(0, 68):
                x = landmarks[n][0]
                y = landmarks[n][1]
                cv2.circle(frame1, (x, y), 2, (0, 255, 0), -1)

            # 计算嘴巴的MAR值
            mouth_mar = get_mar(mouth_landmarks, landmarks)
            # 判断嘴巴是否张开，并记录当前帧的嘴巴状态
            if mouth_mar > mouth_open_threshold:
                mouth_state = 1  # 嘴巴张开
            else:
                mouth_state = 0  # 嘴巴闭合
            # 如果当前帧的嘴巴状态和最近几帧的平均状态不同，就显示或者隐藏提醒
            if mouth_state != sum(last_mouth_state) / len(last_mouth_state):
                if mouth_state == 1:
                    font_path = "simhei.ttf"  # 这里你可以用其他的中文字体文件
                    font = ImageFont.truetype(font_path, 50)  # 这里你可以调整字体大小
                    pil_img = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))  # 转换颜色空间
                    draw = ImageDraw.Draw(pil_img)  # 创建一个画笔对象
                    draw.text((20, 120), "打哈欠！", font=font, fill=(255, 5, 5))  # 在图像上写中文
                    frame1 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 把PIL的图像转换回OpenCV的图像
                else:
                    cv2.putText(frame1, " ", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # 这里用空白文本覆盖提醒

            # 更新最近几帧的嘴巴状态
            last_mouth_state.pop(0)  # 移除最早的一帧
            last_mouth_state.append(mouth_state)  # 添加当前帧

        q.put(frame1)

def program2():
    while True:
        # 从队列中获取图像
        frame = q.get()
        # 将图像从BGR格式转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将图像转换为PIL格式，并缩放到640x640大小，这样可以加速模型的预测
        pil_frame = Image.fromarray(rgb_frame).resize((640,640))

        # 将图像转换为numpy数组，并归一化到[0,1]范围，这样可以符合模型的输入要求
        img = np.array(pil_frame).astype(np.float32) / 255.0
        # 将图像的形状从[H,W,C]转换为[1,C,H,W]，这样可以符合模型的输入要求
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        # 使用onnx会话来进行预测，并获取检测结果
        outputs = session.run(None, {'images': img})
        detections = outputs[0]

        # 在原始图像上绘制预测结果
        for *xyxy, conf, cls in detections:
            # 如果类别为phone
            if cls == 2:
                # 如果置信度大于0.5
                if conf > 0.5:
                    # 在图像上绘制提醒信息
                    pil_frame = Image.fromarray(rgb_frame)
                    draw = ImageDraw.Draw(pil_frame)
                    font = ImageFont.truetype('simhei.ttf', 32)
                    draw.text((250, 250), '正在使用手机', fill=(0, 0, 255), font=font)
                    rgb_frame = np.array(pil_frame)
            # 如果类别为smoke
            elif cls == 1:
                # 如果置信度大于0.5
                if conf > 0.5:
                    # 在图像上绘制提醒信息
                    pil_frame = Image.fromarray(rgb_frame)
                    draw = ImageDraw.Draw(pil_frame)
                    font = ImageFont.truetype('simhei.ttf', 32)
                    draw.text((300, 300), '正在吸烟', fill=(0, 255, 0), font=font)
                    rgb_frame = np.array(pil_frame)
            # 如果类别为drink
            elif cls == 3:
                # 如果置信度大于0.5
                if conf > 0.5:
                    # 在图像上绘制提醒信息
                    pil_frame = Image.fromarray(rgb_frame)
                    draw = ImageDraw.Draw(pil_frame)
                    font = ImageFont.truetype('simhei.ttf', 32)
                    draw.text((350, 350), '正在喝水', fill=(255, 0, 0), font=font)
                    rgb_frame = np.array(pil_frame)

        # 将图像从RGB格式转换为BGR格式，并显示在窗口中
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame1', bgr_frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def processImg(frame):
    # 对图像进行一些处理，这里使用numba库来加速图像处理的速度
    from numba import jit

    @jit(nopython=True) # 使用jit装饰器来加速函数的运行
    def process(frame):
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测
        gray = cv2.Canny(gray, 50, 150)
        return gray

    global bgr_frame # 声明一个全局变量，用来存储处理后的图像
    bgr_frame = process(frame) # 调用process函数来处理图像

q = Queue()
t1 = threading.Thread(target=program1)
t2 = threading.Thread(target=program2)

t1.start()
t2.start()

while True:
    # 从队列中获取图像
    frame = q.get()

    # 在同一个窗口中显示图像
    cv2.imshow('frame', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

t1.join()
t2.join()
# 如果你在程序中使用了 cv2.VideoCapture 来捕获图像，那么应该在这里释放资源
cap.release()

cv2.destroyAllWindows()