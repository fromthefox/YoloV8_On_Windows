# YOLOv8 配置文件
# 您可以修改以下参数来自定义检测效果

# 模型设置
model_name = "yolov8n.pt"  # 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
device = "cpu"  # 设备类型，目前只支持cpu

# 摄像头设置
camera_id = 0  # 摄像头ID，0为默认摄像头，如果有多个摄像头可以尝试1、2等
frame_width = 640  # 摄像头分辨率宽度
frame_height = 480  # 摄像头分辨率高度
fps = 30  # 摄像头帧率

# 检测设置
confidence_threshold = 0.5  # 置信度阈值，0.0-1.0，越高越严格
max_detections = 100  # 最大检测数量

# 显示设置
show_fps = True  # 是否显示FPS
show_count = True  # 是否显示检测数量
font_scale = 1.0  # 字体大小
line_thickness = 2  # 边框线条粗细

# 保存设置
save_format = "jpg"  # 保存图像格式
save_quality = 95  # 保存图像质量 (1-100)

# 性能设置
skip_frames = 0  # 跳过帧数，用于提高性能，0表示不跳过

# 多窗口分类设置
# 只有在此处指定的类别会被检测并显示，其他类别即使被检测到也不会显示
window_categories = {
    "window1": ["person", "handbag", "backpack", "suitcase"],  # 第一个窗口：人、火车
    "window2": ["keyboard", "cell phone"],  # 第二个窗口：手机、杯子、汽车
    "window3": ["laptop", "tv"]  # 第三个窗口：电脑、键盘等
}

# 窗口显示设置
window_titles = {
    "window1": "Window 1 - Person & Train",
    "window2": "Window 2 - Phone & Cup & Car", 
    "window3": "Window 3 - Computer & Keyboard"
}

# 窗口位置设置 (x, y)
window_positions = {
    "window1": (50, 50),
    "window2": (700, 50),
    "window3": (350, 400)
}

# 独占显示设置
exclusive_display = True  # 是否启用独占显示模式
# 独占显示逻辑：
# - 每个窗口独立判断：如果该窗口检测到对应类别就显示，没有检测到就黑屏
# - 多个窗口可以同时显示检测结果
# - 只有当某个窗口没有检测到任何指定类别时才显示黑屏
exclusive_priority = ["window1", "window2", "window3"]  # 优先级顺序（保留兼容性，实际不再使用）
black_screen_text = "No Detection"  # 黑屏时显示的文本
show_waiting_message = True  # 是否在黑屏时显示等待消息
