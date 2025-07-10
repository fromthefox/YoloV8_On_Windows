#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 启动器
提供菜单选项，让用户选择不同的运行模式
"""

import os
import sys
import subprocess

def run_command(command):
    """运行命令"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    """主菜单"""
    print("=" * 60)
    print("🚀 YOLOv8 实时目标检测系统")
    print("=" * 60)
    print("请选择要运行的程序:")
    print("1. 🎯 运行实时目标检测 (主程序)")
    print("2. 🔧 环境测试")
    print("3. 📋 查看环境信息")
    print("4. 📖 查看使用说明")
    print("5. 🚪 退出")
    print("=" * 60)
    
    while True:
        try:
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == "1":
                print("\n🎯 启动实时目标检测...")
                print("按 'q' 键退出，按 's' 键保存图像")
                print("-" * 40)
                
                # 运行主程序
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python yolo_webcam.py'
                success, stdout, stderr = run_command(cmd)
                
                if not success:
                    print(f"❌ 运行失败: {stderr}")
                else:
                    print("✅ 程序运行完成")
                    
            elif choice == "2":
                print("\n🔧 运行环境测试...")
                print("-" * 40)
                
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python test_yolo.py'
                success, stdout, stderr = run_command(cmd)
                
                if success:
                    print("✅ 环境测试完成")
                else:
                    print(f"❌ 测试失败: {stderr}")
                    
            elif choice == "3":
                print("\n📋 环境信息:")
                print("-" * 40)
                
                # 显示Python版本
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python --version'
                success, stdout, stderr = run_command(cmd)
                if success:
                    print(f"Python版本: {stdout.strip()}")
                
                # 显示主要包版本
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python -c "import torch; print(f\'PyTorch: {torch.__version__}\'); import cv2; print(f\'OpenCV: {cv2.__version__}\'); import numpy; print(f\'NumPy: {numpy.__version__}\'); from ultralytics import YOLO; print(\'Ultralytics: 已安装\')"'
                success, stdout, stderr = run_command(cmd)
                if success:
                    print(stdout.strip())
                
                print(f"当前工作目录: {os.getcwd()}")
                
            elif choice == "4":
                print("\n📖 使用说明:")
                print("-" * 40)
                print("1. 实时检测:")
                print("   - 按 'q' 键退出程序")
                print("   - 按 's' 键保存当前帧")
                print("   - 左上角显示FPS和检测对象数量")
                print("")
                print("2. 支持的检测类别:")
                print("   - 人物、车辆、动物、日常用品等80个类别")
                print("")
                print("3. 性能优化:")
                print("   - 使用CPU推理，无需GPU")
                print("   - 自动调整摄像头分辨率")
                print("")
                print("4. 故障排除:")
                print("   - 如果摄像头无法打开，请检查是否被其他程序占用")
                print("   - 确保在正确的conda环境中运行")
                
            elif choice == "5":
                print("\n👋 谢谢使用！")
                break
                
            else:
                print("❌ 无效选择，请输入1-5之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
