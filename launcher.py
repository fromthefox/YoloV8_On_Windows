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
    print("1. 🎯 运行三窗口实时检测 (新功能)")
    print("2. 🎮 三窗口演示模式")
    print("3. 🔧 环境测试")
    print("4. 📋 查看环境信息")
    print("5. 📖 查看使用说明")
    print("6. 🚪 退出")
    print("=" * 60)
    
    while True:
        try:
            choice = input("请输入选择 (1-6): ").strip()
            
            if choice == "1":
                print("\n🎯 启动三窗口实时检测...")
                print("✨ 三窗口分类说明:")
                print("  窗口1: 人物、火车")
                print("  窗口2: 手机、杯子、汽车")
                print("  窗口3: 电脑、键盘、电视、鼠标、遥控器")
                print("  其他类别: 随机分配")
                print("按 'q' 键退出，按 's' 键保存所有窗口，按 '1'/'2'/'3' 切换窗口焦点")
                print("-" * 40)
                
                # 运行三窗口检测
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python yolo_webcam.py'
                success, stdout, stderr = run_command(cmd)
                
                if not success:
                    print(f"❌ 运行失败: {stderr}")
                else:
                    print("✅ 三窗口检测完成")
                    
            elif choice == "2":
                print("\n🎮 启动三窗口演示模式...")
                print("-" * 40)
                
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python demo_multi_window.py'
                success, stdout, stderr = run_command(cmd)
                
                if success:
                    print("✅ 演示完成")
                else:
                    print(f"❌ 演示失败: {stderr}")
                    
            elif choice == "3":
                print("\n🔧 运行环境测试...")
                print("-" * 40)
                
                cmd = 'D:/Anaconda/Scripts/conda.exe run -p C:\\Users\\yhbia\\.conda\\envs\\yolo_test --no-capture-output python test_yolo.py'
                success, stdout, stderr = run_command(cmd)
                
                if success:
                    print("✅ 环境测试完成")
                else:
                    print(f"❌ 测试失败: {stderr}")
                    
            elif choice == "4":
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
                
            elif choice == "5":
                print("\n📖 使用说明:")
                print("-" * 40)
                print("1. 三窗口实时检测:")
                print("   - 按 'q' 键退出程序")
                print("   - 按 's' 键保存所有窗口的当前帧")
                print("   - 按 '1', '2', '3' 键切换窗口焦点")
                print("   - 每个窗口显示不同类别的检测结果")
                print("")
                print("2. 窗口分类说明:")
                print("   - 窗口1: 人物(person)、火车(train)")
                print("   - 窗口2: 手机(cell phone)、杯子(cup)、汽车(car)")
                print("   - 窗口3: 电脑(laptop)、键盘(keyboard)、电视(tv)、鼠标(mouse)、遥控器(remote)")
                print("   - 其他类别: 随机分配到三个窗口")
                print("")
                print("3. 支持的检测类别:")
                print("   - 人物、车辆、动物、日常用品等80个类别")
                print("")
                print("4. 性能优化:")
                print("   - 使用CPU推理，无需GPU")
                print("   - 可通过config.py调整参数")
                print("")
                print("5. 故障排除:")
                print("   - 如果窗口位置重叠，请修改config.py中的window_positions")
                print("   - 如果摄像头无法打开，请检查camera_id设置")
                print("   - 确保在正确的conda环境中运行")
                
            elif choice == "6":
                print("\n👋 谢谢使用！")
                break
                
            else:
                print("❌ 无效选择，请输入1-6之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
