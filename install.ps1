# YOLOv8 环境安装脚本
# 请在激活 yolo_test 环境后运行此脚本

Write-Host "=" -ForegroundColor Green -NoNewline
Write-Host "=" * 50 -ForegroundColor Green
Write-Host "YOLOv8 环境安装脚本" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

# 检查是否在正确的conda环境中
$env_name = conda info --envs | Select-String -Pattern "\*" | ForEach-Object { $_.Line.Split()[0] }
Write-Host "当前环境: $env_name" -ForegroundColor Yellow

if ($env_name -ne "yolo_test") {
    Write-Host "警告: 当前不在 yolo_test 环境中" -ForegroundColor Red
    Write-Host "请先运行: conda activate yolo_test" -ForegroundColor Red
    Read-Host "按回车键继续或 Ctrl+C 退出"
}

# 安装PyTorch (CPU版本)
Write-Host "`n正在安装PyTorch CPU版本..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
Write-Host "`n正在安装其他依赖包..." -ForegroundColor Cyan
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pillow
pip install matplotlib

# 验证安装
Write-Host "`n验证安装..." -ForegroundColor Cyan
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import cv2; print('OpenCV版本:', cv2.__version__)"
python -c "import numpy; print('NumPy版本:', numpy.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics安装成功')"

Write-Host "`n安装完成！" -ForegroundColor Green
Write-Host "现在可以运行 python test_yolo.py 进行测试" -ForegroundColor Green
