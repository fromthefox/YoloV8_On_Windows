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
    # 多窗口设置
    WINDOW_CATEGORIES = config.window_categories
    WINDOW_TITLES = config.window_titles
    WINDOW_POSITIONS = config.window_positions
    # 独占显示设置
    EXCLUSIVE_DISPLAY = config.exclusive_display
    EXCLUSIVE_PRIORITY = config.exclusive_priority
    BLACK_SCREEN_TEXT = config.black_screen_text
    SHOW_WAITING_MESSAGE = config.show_waiting_message
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
    # 多窗口设置（默认）
    WINDOW_CATEGORIES = {
        "window1": ["person", "train"],
        "window2": ["cell phone", "cup", "car"],
        "window3": ["laptop", "keyboard", "tv", "mouse", "remote"]
    }
    WINDOW_TITLES = {
        "window1": "Window 1 - Person & Train",
        "window2": "Window 2 - Phone, Cup & Car", 
        "window3": "Window 3 - Computer Devices"
    }
    WINDOW_POSITIONS = {
        "window1": (100, 100),
        "window2": (750, 100),
        "window3": (1400, 100)
    }
    # 独占显示设置（默认）
    EXCLUSIVE_DISPLAY = True
    EXCLUSIVE_PRIORITY = ["window1", "window2", "window3"]
    BLACK_SCREEN_TEXT = "No detections"
    SHOW_WAITING_MESSAGE = True
    print("⚠️ 未找到配置文件，使用默认设置")

class YOLODetector:
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
        
        # 为随机分配创建所有类别的列表
        self.all_categories = list(WINDOW_CATEGORIES.values())
        self.all_categories_flat = [item for sublist in self.all_categories for item in sublist]
        
    def categorize_detection(self, class_name):
        """
        根据类别名称确定应该显示在哪个窗口
        
        Args:
            class_name (str): 检测到的类别名称
            
        Returns:
            str or None: 窗口名称 (window1, window2, window3) 或 None (不显示)
        """
        # 检查是否属于特定窗口的类别
        for window_name, categories in WINDOW_CATEGORIES.items():
            if class_name in categories:
                return window_name
        
        # 如果不在指定类别中，返回None（不显示）
        return None
        
    def detect_multi_window(self, frame):
        """
        对单帧图像进行目标检测并分类到不同窗口
        
        Args:
            frame: 输入图像帧
            
        Returns:
            dict: 包含三个窗口检测结果的字典
        """
        # 进行推理
        results = self.model(frame, device=self.device, verbose=False)
        
        # 初始化三个窗口的检测结果
        window_results = {
            "window1": {"detections": [], "frame": frame.copy()},
            "window2": {"detections": [], "frame": frame.copy()},
            "window3": {"detections": [], "frame": frame.copy()}
        }
        
        if results and len(results) > 0:
            result = results[0]
            
            # 获取检测框
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                confidences = result.boxes.conf.cpu().numpy()  # 置信度
                class_ids = result.boxes.cls.cpu().numpy()  # 类别ID
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf > self.confidence_threshold:
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.class_names[int(cls_id)]
                        
                        # 确定应该显示在哪个窗口
                        target_window = self.categorize_detection(class_name)
                        
                        # 只处理在指定类别中的检测结果
                        if target_window is not None:
                            # 保存检测结果
                            detection_info = {
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'color': self.colors[int(cls_id) % len(self.colors)]
                            }
                            
                            window_results[target_window]["detections"].append(detection_info)
                            
                            # 在对应窗口的图像上绘制检测框
                            self.draw_detection(window_results[target_window]["frame"], detection_info)
        
        return window_results
        
    def draw_detection(self, frame, detection_info):
        """
        在图像上绘制检测结果
        
        Args:
            frame: 图像帧
            detection_info: 检测信息字典
        """
        x1, y1, x2, y2 = detection_info['bbox']
        color = detection_info['color']
        class_name = detection_info['class']
        conf = detection_info['confidence']
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
        
        # 绘制标签
        label = f'{class_name}: {conf:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, LINE_THICKNESS)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, (255, 255, 255), LINE_THICKNESS)
        
    def create_black_screen(self, width, height, window_name):
        """
        创建黑屏图像
        
        Args:
            width: 图像宽度
            height: 图像高度
            window_name: 窗口名称
            
        Returns:
            black_screen: 黑屏图像
        """
        # 创建黑色背景
        black_screen = np.zeros((height, width, 3), dtype=np.uint8)
        
        if SHOW_WAITING_MESSAGE:
            # 添加等待消息
            window_id = window_name[-1]
            
            # 主要文本
            main_text = f"Window {window_id}"
            text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2 - 60
            cv2.putText(black_screen, main_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            
            # 状态文本
            status_text = BLACK_SCREEN_TEXT
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2 - 20
            cv2.putText(black_screen, status_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
            
            # 等待图标 (简单的圆点)
            center_x = width // 2
            center_y = height // 2 + 30
            cv2.circle(black_screen, (center_x, center_y), 10, (60, 60, 60), -1)
            
            # 显示当前时间
            import time
            current_time = time.strftime("%H:%M:%S")
            time_text = f"Time: {current_time}"
            text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 30
            cv2.putText(black_screen, time_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        
        return black_screen
        
    def determine_active_windows(self, window_results):
        """
        确定哪些窗口应该显示（独占显示模式）
        
        Args:
            window_results: 各窗口的检测结果
            
        Returns:
            set: 应该显示的窗口名称集合
        """
        if not EXCLUSIVE_DISPLAY:
            return {"window1", "window2", "window3"}  # 非独占模式，所有窗口都显示
        
        # 独占模式：只显示有检测结果的窗口，没有检测的窗口显示黑屏
        active_windows = set()
        for window_name, result in window_results.items():
            if len(result["detections"]) > 0:
                active_windows.add(window_name)
        
        return active_windows
        
    def detect(self, frame):
        """
        对单帧图像进行目标检测
        
        Args:
            frame: 输入图像帧
            
        Returns:
            tuple: (检测结果列表, 标注后的图像)
        """
        # 进行推理
        results = self.model(frame, device=self.device, verbose=False)
        
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
                    if conf > self.confidence_threshold:
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.class_names[int(cls_id)]
                        color = self.colors[int(cls_id) % len(self.colors)]
                        
                        # 保存检测结果
                        detection_info = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'color': color
                        }
                        detections.append(detection_info)
                        
                        # 在图像上绘制检测框
                        self.draw_detection(annotated_frame, detection_info)
        
        return detections, annotated_frame
    
    def run_webcam(self):
        """
        运行摄像头实时检测（多窗口版本）
        """
        # 初始化摄像头
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        print("📹 摄像头已启动")
        print("🎯 三窗口实时检测已开始...")
        print(f"📊 窗口1类别: {WINDOW_CATEGORIES['window1']}")
        print(f"📊 窗口2类别: {WINDOW_CATEGORIES['window2']}")
        print(f"📊 窗口3类别: {WINDOW_CATEGORIES['window3']}")
        print("\n⌨️ 控制键:")
        print("  Q - 退出程序")
        print("  S - 保存当前帧到文件")
        print("  C - 清除控制台显示检测统计")
        print("  E - 切换独占/普通显示模式")
        print("  1/2/3 - 将对应窗口置顶")
        print("  + - 增加置信度阈值")
        print("  - - 减少置信度阈值")
        print()
        
        # 创建窗口并设置位置
        for window_name, title in WINDOW_TITLES.items():
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, FRAME_WIDTH, FRAME_HEIGHT)
            if window_name in WINDOW_POSITIONS:
                x, y = WINDOW_POSITIONS[window_name]
                cv2.moveWindow(title, x, y)
        
        # 性能监控变量
        prev_time = time.time()
        frame_count = 0
        skip_counter = 0
        
        # 统计变量
        total_detections = {"window1": 0, "window2": 0, "window3": 0}
        
        # 独占显示模式的动态变量
        current_exclusive_mode = EXCLUSIVE_DISPLAY
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头数据")
                    break
                
                # 跳帧处理（提高性能）
                if SKIP_FRAMES > 0:
                    skip_counter += 1
                    if skip_counter <= SKIP_FRAMES:
                        continue
                    skip_counter = 0
                    
                # 进行多窗口目标检测
                window_results = self.detect_multi_window(frame)
                
                # 确定活跃窗口（独占显示模式）
                active_windows = self.determine_active_windows(window_results)
                
                # 计算FPS
                current_time = time.time()
                frame_count += 1
                if current_time - prev_time >= 1.0:
                    fps = frame_count / (current_time - prev_time)
                    prev_time = current_time
                    frame_count = 0
                else:
                    fps = 0
                
                # 在每个窗口上添加信息并显示
                for window_name, result in window_results.items():
                    # 判断当前窗口是否应该显示实际图像
                    if current_exclusive_mode:
                        if window_name in active_windows:
                            # 显示检测结果
                            annotated_frame = result["frame"]
                            detections = result["detections"]
                        else:
                            # 显示黑屏
                            annotated_frame = self.create_black_screen(FRAME_WIDTH, FRAME_HEIGHT, window_name)
                            detections = []
                    else:
                        # 非独占模式，显示所有窗口
                        annotated_frame = result["frame"]
                        detections = result["detections"]
                    
                    # 添加窗口信息
                    info_y = 30
                    
                    # 显示FPS
                    if SHOW_FPS and fps > 0:
                        color = (0, 255, 0) if not current_exclusive_mode or window_name in active_windows else (100, 100, 100)
                        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, info_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, color, LINE_THICKNESS)
                        info_y += 30
                    
                    # 显示当前窗口检测数量
                    if SHOW_COUNT:
                        color = (0, 255, 0) if not current_exclusive_mode or window_name in active_windows else (100, 100, 100)
                        cv2.putText(annotated_frame, f'Objects: {len(detections)}', (10, info_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, color, LINE_THICKNESS)
                        info_y += 30
                    
                    # 显示窗口标识
                    window_id = window_name[-1]  # 获取窗口编号
                    if current_exclusive_mode and window_name in active_windows:
                        color = (0, 255, 255)  # 活跃窗口用亮色
                        status = f'Window {window_id} [ACTIVE]'
                    elif current_exclusive_mode:
                        color = (100, 100, 100)  # 非活跃窗口用暗色
                        status = f'Window {window_id} [WAITING]'
                    else:
                        color = (255, 255, 0)  # 普通模式用黄色
                        status = f'Window {window_id}'
                    
                    cv2.putText(annotated_frame, status, (10, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, color, LINE_THICKNESS)
                    
                    # 显示独占模式状态
                    mode_text = "EXCLUSIVE" if current_exclusive_mode else "NORMAL"
                    cv2.putText(annotated_frame, f'Mode: {mode_text}', (10, annotated_frame.shape[0] - 40),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, (255, 255, 255), LINE_THICKNESS)
                    
                    # 显示置信度阈值
                    cv2.putText(annotated_frame, f'Confidence: {self.confidence_threshold:.2f}', 
                              (10, annotated_frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.5, (0, 255, 0), LINE_THICKNESS)
                    
                    # 更新总检测数
                    total_detections[window_name] = len(result["detections"])  # 使用原始检测数
                    
                    # 显示图像
                    cv2.imshow(WINDOW_TITLES[window_name], annotated_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存所有窗口的当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    for window_name, result in window_results.items():
                        window_id = window_name[-1]
                        filename = f'detection_window{window_id}_{timestamp}.{SAVE_FORMAT}'
                        # 根据模式保存对应的图像
                        if current_exclusive_mode and window_name in active_windows:
                            save_frame = result["frame"]
                        elif current_exclusive_mode:
                            save_frame = self.create_black_screen(FRAME_WIDTH, FRAME_HEIGHT, window_name)
                        else:
                            save_frame = result["frame"]
                            
                        if SAVE_FORMAT.lower() == 'jpg':
                            cv2.imwrite(filename, save_frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_QUALITY])
                        else:
                            cv2.imwrite(filename, save_frame)
                    print(f"📸 已保存所有窗口图像 ({timestamp})")
                elif key == ord('c'):
                    # 清除控制台
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("🎯 三窗口实时检测运行中...")
                    if current_exclusive_mode:
                        active_list = ', '.join(active_windows) if active_windows else '无'
                        print(f"🔥 独占模式: 当前活跃窗口 - {active_list}")
                    print(f"📊 当前检测统计: W1={total_detections['window1']}, W2={total_detections['window2']}, W3={total_detections['window3']}")
                elif key == ord('e'):
                    # 切换独占显示模式
                    current_exclusive_mode = not current_exclusive_mode
                    mode_name = "独占模式" if current_exclusive_mode else "普通模式"
                    print(f"🔄 已切换到{mode_name}")
                elif key == ord('1'):
                    #  查询模型1服务节点
                    cv2.setWindowProperty(WINDOW_TITLES["window1"], cv2.WND_PROP_TOPMOST, 1)
                    print("🎯 切换到窗口1焦点")
                elif key == ord('2'):
                    #  查询模型2服务节点
                    cv2.setWindowProperty(WINDOW_TITLES["window2"], cv2.WND_PROP_TOPMOST, 1)
                    print("🎯 切换到窗口2焦点")
                elif key == ord('3'):
                    #  查询模型3服务节点
                    cv2.setWindowProperty(WINDOW_TITLES["window3"], cv2.WND_PROP_TOPMOST, 1)
                    print("🎯 切换到窗口3焦点")
                elif key == ord('+') or key == ord('='):
                    # 增加置信度阈值
                    if self.confidence_threshold < 0.95:
                        self.confidence_threshold += 0.05
                        print(f"📈 置信度阈值增加到: {self.confidence_threshold:.2f}")
                elif key == ord('-'):
                    # 减少置信度阈值
                    if self.confidence_threshold > 0.1:
                        self.confidence_threshold -= 0.05
                        print(f"📉 置信度阈值减少到: {self.confidence_threshold:.2f}")
                        
        except KeyboardInterrupt:
            print("\n🛑 用户中断程序")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            print("✓ 资源已清理")

def main():
    """
    主函数
    """
    try:
        print("🚀 YOLOv8多窗口目标检测系统")
        print("=" * 50)
        
        # 创建检测器
        detector = YOLODetector()
        
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
