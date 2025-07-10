# YOLOv8 配置文件
# 您可以修改以下参数来自定义检测效果

# 模型设置
model_name = "yolov8n.pt"  # 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
device = "cpu"  # 设备类型，目前只支持cpu

# 摄像头设置
camera_id = 1  # 摄像头ID，0为默认摄像头，如果有多个摄像头可以尝试1、2等
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
