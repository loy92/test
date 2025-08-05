#!/usr/bin/env python3
"""
启动完整的PBR材质识别系统
包含真实的计算机视觉分析功能
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def main():
    """主函数"""
    print("🚀 启动完整PBR材质识别系统")
    print("=" * 50)
    print("✨ 使用真实的计算机视觉分析")
    print("🔬 基于深度学习和传统CV算法")
    print("=" * 50)
    
    # 检查依赖
    try:
        import torch
        import cv2
        import numpy
        import sklearn
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("正在尝试安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "torch", "opencv-python", "scikit-image"], check=True)
            print("✅ 依赖安装完成")
        except:
            print("❌ 依赖安装失败，请手动安装")
            return
    
    # 启动API服务器
    print("\n正在启动API服务器...")
    api_process = subprocess.Popen([
        sys.executable, "api/app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待API启动
    time.sleep(3)
    
    # 启动Web服务器
    print("正在启动Web服务器...")
    web_dir = Path("web")
    web_process = subprocess.Popen([
        sys.executable, "-m", "http.server", "8081"
    ], cwd=web_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)
    
    # 测试API
    try:
        import requests
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ API服务器运行正常")
        else:
            print("⚠️ API服务器可能有问题")
    except:
        print("⚠️ 无法连接API服务器")
    
    # 打开浏览器
    try:
        webbrowser.open("http://localhost:8081")
        print("✅ 浏览器已打开")
    except:
        print("请手动访问: http://localhost:8081")
    
    print("\n🎉 系统启动完成！")
    print("🌐 Web界面: http://localhost:8081")
    print("🔧 API接口: http://localhost:5000")
    print("\n📊 现在支持真实的PBR分析:")
    print("  • 基于计算机视觉的材质检测")
    print("  • 金属度、粗糙度、透明度分析")
    print("  • 材质数据库参考修正")
    print("  • 高置信度结果输出")
    print("\n按 Ctrl+C 停止系统")
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭系统...")
        api_process.terminate()
        web_process.terminate()
        api_process.wait()
        web_process.wait()
        print("✅ 系统已关闭")


if __name__ == '__main__':
    main() 