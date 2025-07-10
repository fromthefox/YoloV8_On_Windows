import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# 尝试加载配置文件
try:
    import config
    # 从配置文件加载设置
    MODEL_NAME = config.model_name
    DEVICE = config.device
    CAMERA_ID = config.camera_id
    FRAME_WIDTH = config.frame_width
    FRAME_HEIGHT = config.frame_height
    FPS = config.fps
    CONFIDENCE_THRESHOLD = config.confidence_threshold
    MAX_DETECTIONS = config.max_detections
    SHOW_FPS = config.show_fps
    SHOW_COUNT = config.show_count
    FONT_SCALE = config.font_scale
    LINE_THICKNESS = config.line_thickness
    SAVE_FORMAT = config.save_format
    SAVE_QUALITY = config.save_quality
    SKIP_FRAMES = config.skip_frames
    print("✓ 配置文件加载成功")
except ImportError:
    # 使用默认设置
    MODEL_NAME = "yolov8n.pt"
    DEVICE = "cpu"
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    CONFIDENCE_THRESHOLD = 0.5
    MAX_DETECTIONS = 100
    SHOW_FPS = True
    SHOW_COUNT = True
    FONT_SCALE = 1.0
    LINE_THICKNESS = 2
    SAVE_FORMAT = "jpg"
    SAVE_QUALITY = 95
    SKIP_FRAMES = 0
    print("⚠ 配置文件未找到，使用默认设置")


class YoloV8Detector:
    def __init__(self, model_path=MODEL_NAME, device=DEVICE):
        """
        初始化YOLOv8检测器
        
        Args:
            model_path (str): YOLO模型文件路径
            device (str): 设备类型，'cpu' 或 'cuda'
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.max_detections = MAX_DETECTIONS
        
        # 加载YOLOv8模型
        print(f"加载YOLOv8模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 设置设备
        if device == 'cpu':
            print("使用CPU进行推理")
        else:
            print(f"使用设备: {device}")
            
        # COCO数据集类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # 颜色列表（BGR格式）
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 20, 147), (0, 128, 128),
            (128, 128, 0), (255, 192, 203), (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 128), (255, 255, 255), (0, 0, 0), (255, 69, 0), (50, 205, 50)
        ]
        
    def detect(self, frame):
        """
        对单帧图像进行目标检测
        
        Args:
            frame: 输入图像帧
            
        Returns:
            tuple: (检测结果, 带标注的图像)
        """
        # 进行推理
        results = self.model(frame, device=self.device, verbose=False)
        
        # 获取检测结果
        detections = []
        annotated_frame = frame.copy()
        
        if results and len(results) > 0:
            result = results[0]
            
            # 获取检测框
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                confidences = result.boxes.conf.cpu().numpy()  # 置信度
                class_ids = result.boxes.cls.cpu().numpy()  # 类别ID
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf > self.confidence_threshold:  # 使用配置的置信度阈值
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.class_names[int(cls_id)]
                        
                        # 保存检测结果
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # 绘制边界框
                        color = self.colors[int(cls_id) % len(self.colors)]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
                        
                        # 绘制标签
                        label = f'{class_name}: {conf:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, LINE_THICKNESS)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, (255, 255, 255), LINE_THICKNESS)
        
        return detections, annotated_frame
        
    def run_webcam(self, camera_id=CAMERA_ID):
        """
        运行摄像头实时检测
        
        Args:
            camera_id (int): 摄像头ID，默认为配置文件中的设置
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
            
        print("摄像头已打开，开始实时检测...")
        print("操作说明:")
        print("- 按 'q' 键退出")
        print("- 按 's' 键保存当前帧")
        print("- 按 'c' 键清除控制台")
        print("=" * 50)
        
        # 初始化FPS计算变量
        prev_time = time.time()
        frame_count = 0
        skip_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头数据")
                    break
                
                # 跳帧处理（提高性能）
                if SKIP_FRAMES > 0:
                    skip_counter += 1
                    if skip_counter <= SKIP_FRAMES:
                        continue
                    skip_counter = 0
                    
                # 进行目标检测
                detections, annotated_frame = self.detect(frame)
                
                # 计算FPS
                current_time = time.time()
                frame_count += 1
                if current_time - prev_time >= 1.0:
                    fps = frame_count / (current_time - prev_time)
                    prev_time = current_time
                    frame_count = 0
                else:
                    fps = 0
                
                # 显示FPS
                if SHOW_FPS and fps > 0:
                    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), LINE_THICKNESS)
                
                # 显示检测数量
                if SHOW_COUNT:
                    cv2.putText(annotated_frame, f'Objects: {len(detections)}', (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), LINE_THICKNESS)
                
                # 显示置信度阈值
                cv2.putText(annotated_frame, f'Confidence: {self.confidence_threshold:.2f}', (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (0, 255, 0), LINE_THICKNESS)
                
                # 显示图像
                cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'detection_{timestamp}.{SAVE_FORMAT}'
                    if SAVE_FORMAT.lower() == 'jpg':
                        cv2.imwrite(filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_QUALITY])
                    else:
                        cv2.imwrite(filename, annotated_frame)
                    print(f"📸 已保存图像: {filename}")
                elif key == ord('c'):
                    # 清除控制台
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("摄像头实时检测运行中...")
                    print("按 'q' 退出, 's' 保存, 'c' 清屏")
                    
        except KeyboardInterrupt:
            print("\n用户中断程序")
            
        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已关闭")


def main():
    """主函数"""
    print("=" * 50)
    print("YOLOv8 实时目标检测系统")
    print("=" * 50)
    
    try:
        # 创建检测器实例
        detector = YoloV8Detector()
        
        # 运行摄像头检测
        detector.run_webcam()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请检查:")
        print("1. 是否正确安装了所需的依赖包")
        print("2. 摄像头是否正常工作")
        print("3. 是否在正确的conda环境中运行")
        input("按回车键退出...")


if __name__ == "__main__":
    main()
