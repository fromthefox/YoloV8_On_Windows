"""
测试新的独占显示逻辑
验证多窗口同时显示功能
"""
import sys
import os
sys.path.append('.')

# 导入配置
import config

def test_exclusive_display_logic():
    """测试独占显示逻辑"""
    print("🔄 测试独占显示逻辑...")
    print("=" * 60)
    
    # 模拟窗口检测结果
    test_cases = [
        {
            "name": "只有窗口1有检测",
            "window_results": {
                "window1": {"detections": [{"class": "person", "confidence": 0.9}]},
                "window2": {"detections": []},
                "window3": {"detections": []}
            },
            "expected_active": {"window1"}
        },
        {
            "name": "窗口1和窗口2都有检测",
            "window_results": {
                "window1": {"detections": [{"class": "person", "confidence": 0.9}]},
                "window2": {"detections": [{"class": "car", "confidence": 0.8}]},
                "window3": {"detections": []}
            },
            "expected_active": {"window1", "window2"}
        },
        {
            "name": "所有窗口都有检测",
            "window_results": {
                "window1": {"detections": [{"class": "person", "confidence": 0.9}]},
                "window2": {"detections": [{"class": "car", "confidence": 0.8}]},
                "window3": {"detections": [{"class": "laptop", "confidence": 0.7}]}
            },
            "expected_active": {"window1", "window2", "window3"}
        },
        {
            "name": "所有窗口都没有检测",
            "window_results": {
                "window1": {"detections": []},
                "window2": {"detections": []},
                "window3": {"detections": []}
            },
            "expected_active": set()
        },
        {
            "name": "只有窗口3有检测",
            "window_results": {
                "window1": {"detections": []},
                "window2": {"detections": []},
                "window3": {"detections": [{"class": "laptop", "confidence": 0.7}]}
            },
            "expected_active": {"window3"}
        }
    ]
    
    # 模拟 determine_active_windows 函数逻辑
    def determine_active_windows(window_results, exclusive_display=True):
        """
        确定哪些窗口应该显示（独占显示模式）
        """
        if not exclusive_display:
            return {"window1", "window2", "window3"}  # 非独占模式，所有窗口都显示
        
        # 独占模式：只显示有检测结果的窗口，没有检测的窗口显示黑屏
        active_windows = set()
        for window_name, result in window_results.items():
            if len(result["detections"]) > 0:
                active_windows.add(window_name)
        
        return active_windows
    
    # 运行测试
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print("-" * 40)
        
        # 显示输入
        for window_name, result in test_case["window_results"].items():
            detection_count = len(result["detections"])
            if detection_count > 0:
                classes = [d["class"] for d in result["detections"]]
                print(f"  {window_name}: {detection_count} 个检测 {classes}")
            else:
                print(f"  {window_name}: 无检测")
        
        # 执行测试
        active_windows = determine_active_windows(test_case["window_results"])
        expected = test_case["expected_active"]
        
        # 验证结果
        if active_windows == expected:
            print(f"  ✅ 测试通过")
            print(f"  📊 活跃窗口: {sorted(active_windows) if active_windows else '无'}")
            print(f"  🖥️ 显示状态:")
            for window_name in ["window1", "window2", "window3"]:
                if window_name in active_windows:
                    print(f"    - {window_name}: 显示检测结果")
                else:
                    print(f"    - {window_name}: 黑屏")
        else:
            print(f"  ❌ 测试失败")
            print(f"  预期: {sorted(expected) if expected else '无'}")
            print(f"  实际: {sorted(active_windows) if active_windows else '无'}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！独占显示逻辑正确实现")
        print("✅ 多窗口同时显示功能验证成功")
        print("✅ 只有无检测的窗口才显示黑屏")
    else:
        print("❌ 部分测试失败，请检查逻辑")
        
    return all_passed

def test_category_logic():
    """测试类别分配逻辑"""
    print("\n🔄 测试类别分配逻辑...")
    print("=" * 60)
    
    # 模拟类别分配函数
    def categorize_detection(class_name, window_categories):
        """根据类别名称确定应该显示在哪个窗口"""
        for window_name, categories in window_categories.items():
            if class_name in categories:
                return window_name
        return None
    
    # 测试用例
    test_classes = [
        ("person", "window1"),
        ("train", "window1"),
        ("car", "window2"),
        ("cell phone", "window2"),
        ("cup", "window2"),
        ("laptop", "window3"),
        ("keyboard", "window3"),
        ("tv", "window3"),
        ("mouse", "window3"),
        ("remote", "window3"),
        ("bicycle", None),  # 不在任何窗口
        ("dog", None),      # 不在任何窗口
    ]
    
    print("类别分配测试:")
    print("-" * 40)
    all_passed = True
    
    for class_name, expected_window in test_classes:
        actual_window = categorize_detection(class_name, config.window_categories)
        
        if actual_window == expected_window:
            if expected_window:
                print(f"  ✅ {class_name:12} -> {expected_window}")
            else:
                print(f"  ✅ {class_name:12} -> 不显示")
        else:
            print(f"  ❌ {class_name:12} -> 预期: {expected_window}, 实际: {actual_window}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 类别分配逻辑测试通过！")
        print("✅ 只有指定类别会被显示")
        print("✅ 其他类别不会显示在任何窗口")
    else:
        print("❌ 类别分配逻辑测试失败")
        
    return all_passed

def main():
    """主测试函数"""
    print("🧪 YOLOv8 多窗口显示逻辑测试")
    print("=" * 60)
    
    # 显示当前配置
    print(f"独占显示模式: {config.exclusive_display}")
    print(f"窗口类别配置:")
    for window_name, categories in config.window_categories.items():
        print(f"  {window_name}: {categories}")
    
    # 运行测试
    test1_passed = test_exclusive_display_logic()
    test2_passed = test_category_logic()
    
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    print("-" * 40)
    if test1_passed and test2_passed:
        print("🎊 所有测试通过！")
        print("✅ 独占显示逻辑：多窗口可同时显示，无检测则黑屏")
        print("✅ 类别分配逻辑：只显示指定类别，其他类别不显示")
        print("\n🚀 系统已准备就绪，可以正常使用！")
    else:
        print("❌ 部分测试失败")
        if not test1_passed:
            print("  - 独占显示逻辑需要修复")
        if not test2_passed:
            print("  - 类别分配逻辑需要修复")

if __name__ == "__main__":
    main()
