#!/usr/bin/env python3
"""
PBR材质识别系统 - 最简启动脚本
仅启动Web界面，使用内置的模拟数据
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def main():
    """主函数"""
    print("🚀 PBR材质识别系统 - 最简启动")
    print("=" * 50)
    
    # 切换到web目录
    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print(f"❌ Web目录不存在: {web_dir}")
        return
    
    print(f"Web目录: {web_dir}")
    
    # 尝试不同的端口
    ports = [8080, 8081, 8082, 3000]
    
    for port in ports:
        try:
            print(f"尝试启动Web服务器 (端口 {port})...")
            
            # 切换到web目录
            os.chdir(web_dir)
            
            # 启动HTTP服务器
            process = subprocess.Popen([
                sys.executable, "-m", "http.server", str(port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待一下看是否启动成功
            time.sleep(2)
            
            # 检查进程是否还在运行
            if process.poll() is None:
                print(f"✅ Web服务器启动成功 (http://localhost:{port})")
                
                # 打开浏览器
                try:
                    webbrowser.open(f"http://localhost:{port}")
                    print("✅ 浏览器已打开")
                except:
                    print(f"请手动访问: http://localhost:{port}")
                
                print("\n🎉 系统启动完成！")
                print(f"Web界面: http://localhost:{port}")
                print("\n注意: 当前仅显示界面，PBR分析功能需要安装完整依赖")
                print("要获得完整功能，请运行: pip3 install torch flask flask-cors numpy pillow")
                print("\n按 Ctrl+C 停止系统")
                
                # 保持运行
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n正在关闭系统...")
                    process.terminate()
                    process.wait()
                    print("✅ 系统已关闭")
                
                return
            
            else:
                print(f"❌ 端口 {port} 启动失败")
                
        except Exception as e:
            print(f"❌ 端口 {port} 启动失败: {e}")
            continue
    
    print("❌ 所有端口都无法使用")
    print("请检查网络设置或尝试手动打开web/index.html文件")

if __name__ == '__main__':
    main() 