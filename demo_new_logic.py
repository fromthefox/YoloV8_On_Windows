"""
演示新的独占显示逻辑
展示多窗口同时显示功能
"""

def demo_exclusive_display():
    """演示独占显示逻辑"""
    print("🎯 YOLOv8 多窗口独占显示逻辑演示")
    print("=" * 60)
    
    scenarios = [
        {
            "title": "场景1: 只检测到人",
            "detections": {
                "window1": ["person"],
                "window2": [],
                "window3": []
            },
            "result": "window1显示检测结果，window2和window3显示黑屏"
        },
        {
            "title": "场景2: 同时检测到人和汽车",
            "detections": {
                "window1": ["person"],
                "window2": ["car"],
                "window3": []
            },
            "result": "window1和window2同时显示检测结果，window3显示黑屏"
        },
        {
            "title": "场景3: 检测到人、汽车和电脑",
            "detections": {
                "window1": ["person"],
                "window2": ["car", "cup"],
                "window3": ["laptop"]
            },
            "result": "所有窗口都显示检测结果，无黑屏"
        },
        {
            "title": "场景4: 没有检测到任何指定类别",
            "detections": {
                "window1": [],
                "window2": [],
                "window3": []
            },
            "result": "所有窗口都显示黑屏"
        },
        {
            "title": "场景5: 只检测到电脑设备",
            "detections": {
                "window1": [],
                "window2": [],
                "window3": ["laptop", "keyboard", "mouse"]
            },
            "result": "只有window3显示检测结果，window1和window2显示黑屏"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 {scenario['title']}")
        print("-" * 40)
        
        # 显示检测情况
        print("🔍 检测情况:")
        for window, detections in scenario['detections'].items():
            if detections:
                print(f"  {window}: {', '.join(detections)}")
            else:
                print(f"  {window}: 无检测")
        
        # 显示结果
        print(f"💻 显示结果: {scenario['result']}")
        
        # 详细状态
        print("🖥️ 详细状态:")
        for window, detections in scenario['detections'].items():
            if detections:
                print(f"  {window}: ✅ 显示检测结果")
            else:
                print(f"  {window}: ⚫ 黑屏")
    
    print("\n" + "=" * 60)
    print("🎉 逻辑优化说明:")
    print("✅ 多窗口可以同时显示检测结果")
    print("✅ 只有当窗口没有检测到对应类别时才显示黑屏")
    print("✅ 不再是只显示一个窗口的独占模式")
    print("✅ 每个窗口独立判断显示或黑屏")

def demo_category_assignment():
    """演示类别分配逻辑"""
    print("\n🔄 类别分配逻辑演示")
    print("=" * 60)
    
    # 窗口类别配置
    window_categories = {
        "window1": ["person", "train"],
        "window2": ["cell phone", "cup", "car"],
        "window3": ["laptop", "keyboard", "tv", "mouse", "remote"]
    }
    
    print("🎯 窗口类别配置:")
    for window, categories in window_categories.items():
        print(f"  {window}: {', '.join(categories)}")
    
    print("\n🔍 检测类别分配示例:")
    print("-" * 40)
    
    # 测试类别
    test_detections = [
        ("person", "显示在window1"),
        ("train", "显示在window1"),
        ("car", "显示在window2"),
        ("cell phone", "显示在window2"),
        ("cup", "显示在window2"),
        ("laptop", "显示在window3"),
        ("keyboard", "显示在window3"),
        ("bicycle", "不显示（不在配置中）"),
        ("dog", "不显示（不在配置中）"),
        ("book", "不显示（不在配置中）")
    ]
    
    for detection, result in test_detections:
        if "不显示" in result:
            print(f"  {detection:12} -> ❌ {result}")
        else:
            print(f"  {detection:12} -> ✅ {result}")
    
    print("\n" + "=" * 60)
    print("🎯 类别分配规则:")
    print("✅ 只有在配置文件中指定的类别才会显示")
    print("✅ 其他类别即使被检测到也不会显示在任何窗口")
    print("✅ 每个类别都有明确的窗口归属")

def main():
    """主函数"""
    print("🎊 YOLOv8 多窗口检测系统逻辑演示")
    print("=" * 60)
    print("本演示展示了优化后的独占显示逻辑和类别分配逻辑")
    
    demo_exclusive_display()
    demo_category_assignment()
    
    print("\n🚀 系统特点总结:")
    print("=" * 60)
    print("1. 🎯 智能多窗口显示：每个窗口独立判断")
    print("2. ⚡ 同时显示多个窗口：不再是传统的独占模式")
    print("3. 🖥️ 智能黑屏：只有无检测的窗口才黑屏")
    print("4. 🔍 选择性显示：只显示配置的类别")
    print("5. 🎨 优化的用户体验：更自然的显示逻辑")
    
    print("\n✨ 使用说明:")
    print("- 运行 python yolo_webcam.py 启动检测")
    print("- 按 'E' 键可以切换独占/普通显示模式")
    print("- 按 'C' 键查看实时检测统计")
    print("- 按 'Q' 键退出程序")

if __name__ == "__main__":
    main()
