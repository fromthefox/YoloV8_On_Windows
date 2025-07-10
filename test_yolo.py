"""
简单的YOLOv8测试脚本
用于验证安装和基本功能
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time


def test_yolo_installation():
    """测试YOLOv8安装是否成功"""
    print("测试YOLOv8安装...")
    
    try:
        # 创建一个简单的测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, 'YOLOv8 Test', (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        # 加载YOLOv8模型
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8模型加载成功")
        
        # 进行一次推理测试
        results = model(test_image, device='cpu', verbose=False)
        print("✓ 推理测试成功")
        
        # 测试摄像头
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ 摄像头可用")
            cap.release()
        else:
            print("✗ 摄像头不可用")
            
        print("\n所有测试通过！可以运行主程序了。")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_webcam_only():
    """仅测试摄像头功能"""
    print("测试摄像头...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return False
        
    print("摄像头测试中，按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头数据")
            break
            
        # 显示当前时间
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Webcam Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    print("=" * 40)
    print("YOLOv8 测试程序")
    print("=" * 40)
    
    choice = input("选择测试类型:\n1. 完整测试\n2. 仅测试摄像头\n请输入选择 (1/2): ")
    
    if choice == "1":
        test_yolo_installation()
    elif choice == "2":
        test_webcam_only()
    else:
        print("无效选择")
