# YOLOv8 实时目标检测系统

这是一个基于YOLOv8的实时目标检测系统，可以调用电脑摄像头进行实时目标检测。

## 系统要求

- Windows 操作系统
- Python 3.9 (通过Conda安装)
- 摄像头设备

## 环境配置

### 1. 创建并激活Conda环境

```bash
# 创建环境
conda create -n yolo_test python=3.9

# 激活环境
conda activate yolo_test
```

### 2. 安装依赖包

#### 方法一：使用安装脚本（推荐）

```powershell
# 在PowerShell中运行
.\install.ps1
```

#### 方法二：手动安装

```bash
# 安装PyTorch CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install ultralytics opencv-python numpy pillow matplotlib
```

#### 方法三：从requirements.txt安装

```bash
pip install -r requirements.txt
```

## 文件说明

- `yolo_webcam.py` - 主程序文件，实现实时目标检测
- `test_yolo.py` - 测试脚本，验证环境配置
- `install.ps1` - PowerShell安装脚本
- `run_yolo.bat` - 快速启动批处理文件
- `requirements.txt` - 依赖包列表

## 使用方法

### 1. 测试环境

首先运行测试脚本验证环境配置：

```bash
python test_yolo.py
```

选择测试类型：
- 选择1：完整测试（包括YOLOv8和摄像头）
- 选择2：仅测试摄像头

### 2. 运行主程序

#### 方法一：使用批处理文件

双击 `run_yolo.bat` 文件

#### 方法二：命令行运行

```bash
# 激活环境
conda activate yolo_test

# 运行程序
python yolo_webcam.py
```

## 操作说明

程序运行后会打开摄像头窗口，显示实时检测结果：

- **退出程序**：按 `q` 键
- **保存图像**：按 `s` 键（会保存当前帧到文件）
- **显示信息**：
  - 左上角显示FPS（每秒帧数）
  - 显示检测到的对象数量
  - 每个检测框显示类别和置信度

## 功能特点

1. **实时检测**：使用摄像头进行实时目标检测
2. **多类别识别**：支持COCO数据集的80个类别
3. **三窗口分类显示**：
   - 窗口1：人、火车
   - 窗口2：手机、杯子、汽车
   - 窗口3：电脑、键盘、电视、鼠标、遥控器
4. **选择性显示**：只显示配置文件中指定的类别，其他类别即使被检测到也不会显示
5. **独占显示模式**：当有检测时，只显示有目标的窗口，其他窗口显示黑屏
6. **CPU推理**：优化的CPU推理，无需GPU
7. **可视化展示**：
   - 彩色边界框
   - 类别标签
   - 置信度分数
   - FPS显示
8. **图像保存**：支持保存检测结果

## 检测类别

系统可以检测以下80个类别的对象：

- 人物：person
- 交通工具：car, bicycle, motorcycle, airplane, bus, train, truck, boat
- 动物：cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, bird
- 日常用品：bottle, cup, fork, knife, spoon, bowl, chair, couch, bed
- 电子产品：tv, laptop, mouse, remote, keyboard, cell phone
- 食物：banana, apple, sandwich, orange, pizza, donut, cake
- 运动用品：sports ball, tennis racket, baseball bat, skateboard, surfboard
- 其他物品：backpack, umbrella, handbag, suitcase, book, clock, vase

## 性能优化

- 使用YOLOv8n（nano）模型，在CPU上运行速度较快
- 摄像头分辨率设置为640x480，平衡性能和质量
- 置信度阈值设置为0.5，减少误检

## 故障排除

### 1. 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 尝试更改摄像头ID（修改代码中的camera_id参数）

### 2. 导入错误
- 确保已正确安装所有依赖包
- 检查是否在正确的conda环境中

### 3. 性能问题
- 降低摄像头分辨率
- 提高置信度阈值
- 关闭其他占用CPU的程序

### 4. 模型下载失败
- 确保网络连接正常
- 首次运行时会自动下载yolov8n.pt模型文件

## 自定义配置

可以通过修改 `yolo_webcam.py` 中的参数来自定义配置：

```python
# 修改模型（可选：yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt）
detector = YoloV8Detector(model_path='yolov8n.pt', device='cpu')

# 修改摄像头ID
detector.run_webcam(camera_id=0)  # 0为默认摄像头

# 修改置信度阈值（在detect方法中）
if conf > 0.5:  # 可调整此值
```

## 许可证

本项目使用MIT许可证。

## 联系信息

如有问题，请联系开发者。
