"""
三窗口检测演示脚本
用于测试和演示三窗口分类检测功能
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# 导入配置
try:
    import config
    print("✓ 配置文件加载成功")
    print("📋 窗口分类设置:")
    for window_name, categories in config.window_categories.items():
        window_id = window_name[-1]
        print(f"  窗口{window_id}: {', '.join(categories)}")
except ImportError:
    print("⚠ 配置文件未找到，请检查config.py文件")

def test_categorization():
    """测试分类逻辑"""
    print("\n🧪 测试分类逻辑:")
    
    # 模拟一些检测结果
    test_classes = [
        "person", "train", "car", "cell phone", "cup", 
        "laptop", "keyboard", "tv", "mouse", "dog", "cat", "bicycle"
    ]
    
    from yolo_webcam import YoloV8Detector
    detector = YoloV8Detector()
    
    for class_name in test_classes:
        window = detector.categorize_detection(class_name)
        window_id = window[-1]
        print(f"  {class_name} -> 窗口{window_id}")

def create_demo_image():
    """创建演示图像"""
    print("\n🎨 创建演示图像...")
    
    # 创建一个演示图像
    demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些文本说明
    cv2.putText(demo_image, "YOLOv8 Multi-Window Detection Demo", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(demo_image, "Window 1: Person, Train", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(demo_image, "Window 2: Phone, Cup, Car", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.putText(demo_image, "Window 3: Laptop, Keyboard, TV", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    cv2.putText(demo_image, "Other objects: Random distribution", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(demo_image, "Press 'q' to exit, 's' to save", (50, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(demo_image, "Press '1', '2', '3' to switch windows", (50, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 绘制一些装饰性的框
    cv2.rectangle(demo_image, (30, 30), (610, 400), (100, 100, 100), 2)
    
    return demo_image

def demo_multi_window():
    """演示多窗口功能"""
    print("\n🎯 启动三窗口演示...")
    
    demo_image = create_demo_image()
    
    # 创建三个窗口
    window_titles = {
        "window1": "Window 1 - Person & Train",
        "window2": "Window 2 - Phone & Cup & Car",
        "window3": "Window 3 - Computer & Keyboard"
    }
    
    window_positions = {
        "window1": (50, 50),
        "window2": (700, 50),
        "window3": (350, 400)
    }
    
    # 创建并设置窗口
    for window_name, title in window_titles.items():
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 640, 480)
        if window_name in window_positions:
            x, y = window_positions[window_name]
            cv2.moveWindow(title, x, y)
    
    print("📱 三个窗口已创建")
    print("🎮 按 'q' 退出演示")
    
    try:
        while True:
            # 在每个窗口显示不同的演示信息
            for window_name, title in window_titles.items():
                window_id = window_name[-1]
                demo_copy = demo_image.copy()
                
                # 添加窗口标识
                cv2.putText(demo_copy, f"This is Window {window_id}", (200, 420),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 添加时间戳
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(demo_copy, f"Time: {timestamp}", (450, 450),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(title, demo_copy)
            
            key = cv2.waitKey(1000) & 0xFF  # 每秒更新一次
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存演示图像
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'demo_multi_window_{timestamp}.jpg'
                cv2.imwrite(filename, demo_image)
                print(f"📸 已保存演示图像: {filename}")
                
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断演示")
    
    finally:
        cv2.destroyAllWindows()
        print("📱 演示窗口已关闭")

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 YOLOv8 三窗口检测演示程序")
    print("=" * 60)
    
    choice = input("选择演示模式:\n1. 测试分类逻辑\n2. 三窗口演示\n3. 启动实时检测\n请输入选择 (1-3): ")
    
    if choice == "1":
        test_categorization()
    elif choice == "2":
        demo_multi_window()
    elif choice == "3":
        print("\n🚀 启动实时三窗口检测...")
        from yolo_webcam import YoloV8Detector
        detector = YoloV8Detector()
        detector.run_webcam()
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
