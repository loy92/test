#!/usr/bin/env python3
"""
启动PBR材质识别系统
同时启动API服务器和Web界面
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
import signal
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class SystemLauncher:
    """系统启动器"""
    
    def __init__(self):
        self.api_process = None
        self.web_process = None
        self.running = False
        
    def start_api_server(self):
        """启动API服务器"""
        print("正在启动API服务器...")
        
        try:
            # 切换到项目根目录
            os.chdir(project_root)
            
            # 启动Flask API服务器
            api_script = project_root / "api" / "app.py"
            self.api_process = subprocess.Popen([
                sys.executable, str(api_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print("✅ API服务器启动成功 (http://localhost:5000)")
            
        except Exception as e:
            print(f"❌ API服务器启动失败: {e}")
            return False
        
        return True
    
    def start_web_server(self):
        """启动Web服务器"""
        print("正在启动Web服务器...")
        
        try:
            # 使用Python内置的HTTP服务器托管静态文件
            web_dir = project_root / "web"
            os.chdir(web_dir)
            
            self.web_process = subprocess.Popen([
                sys.executable, "-m", "http.server", "8080"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print("✅ Web服务器启动成功 (http://localhost:8080)")
            
        except Exception as e:
            print(f"❌ Web服务器启动失败: {e}")
            return False
        
        return True
    
    def check_dependencies(self):
        """检查依赖"""
        print("正在检查系统依赖...")
        
        missing_deps = []
        
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
        except ImportError:
            missing_deps.append("torch")
        
        try:
            import flask
            print(f"✅ Flask: {flask.__version__}")
        except ImportError:
            missing_deps.append("flask")
        
        try:
            import PIL
            print(f"✅ Pillow: {PIL.__version__}")
        except ImportError:
            missing_deps.append("Pillow")
        
        try:
            import numpy
            print(f"✅ NumPy: {numpy.__version__}")
        except ImportError:
            missing_deps.append("numpy")
        
        if missing_deps:
            print(f"❌ 缺少依赖包: {', '.join(missing_deps)}")
            print("请运行: pip install -r requirements.txt")
            return False
        
        print("✅ 所有依赖检查通过")
        return True
    
    def wait_for_server(self, url: str, timeout: int = 30):
        """等待服务器启动"""
        import requests
        
        for _ in range(timeout):
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def open_browser(self):
        """打开浏览器"""
        print("正在打开浏览器...")
        
        try:
            webbrowser.open("http://localhost:8080")
            print("✅ 浏览器已打开")
        except Exception as e:
            print(f"❌ 无法自动打开浏览器: {e}")
            print("请手动访问: http://localhost:8080")
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print("\n正在关闭系统...")
        self.stop()
        sys.exit(0)
    
    def stop(self):
        """停止所有服务"""
        self.running = False
        
        if self.api_process:
            print("正在停止API服务器...")
            self.api_process.terminate()
            self.api_process.wait()
        
        if self.web_process:
            print("正在停止Web服务器...")
            self.web_process.terminate()
            self.web_process.wait()
        
        print("✅ 所有服务已停止")
    
    def start(self):
        """启动系统"""
        print("🚀 PBR材质识别系统启动器")
        print("=" * 50)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 启动API服务器
        if not self.start_api_server():
            return False
        
        # 等待API服务器启动
        time.sleep(3)
        
        # 启动Web服务器
        if not self.start_web_server():
            self.stop()
            return False
        
        # 等待Web服务器启动
        time.sleep(2)
        
        # 打开浏览器
        self.open_browser()
        
        print("\n🎉 系统启动完成！")
        print("API服务器: http://localhost:5000")
        print("Web界面: http://localhost:8080")
        print("\n按 Ctrl+C 停止系统")
        
        # 保持运行
        self.running = True
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        return True


def create_desktop_shortcut():
    """创建桌面快捷方式"""
    try:
        import platform
        
        if platform.system() == "Windows":
            # Windows快捷方式
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "PBR材质识别系统.bat")
            
            with open(shortcut_path, 'w', encoding='utf-8') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{project_root}"\n')
                f.write(f'python scripts/start_system.py\n')
                f.write(f'pause\n')
            
            print(f"✅ 已创建桌面快捷方式: {shortcut_path}")
            
        elif platform.system() == "Darwin":  # macOS
            # macOS快捷方式
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "PBR材质识别系统.command")
            
            with open(shortcut_path, 'w') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f'cd "{project_root}"\n')
                f.write(f'python3 scripts/start_system.py\n')
            
            os.chmod(shortcut_path, 0o755)
            print(f"✅ 已创建桌面快捷方式: {shortcut_path}")
            
        else:  # Linux
            # Linux快捷方式
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "PBR材质识别系统.desktop")
            
            with open(shortcut_path, 'w') as f:
                f.write(f'[Desktop Entry]\n')
                f.write(f'Name=PBR材质识别系统\n')
                f.write(f'Comment=基于Nerfies的PBR材质参数识别\n')
                f.write(f'Exec=python3 "{project_root}/scripts/start_system.py"\n')
                f.write(f'Path={project_root}\n')
                f.write(f'Icon=applications-science\n')
                f.write(f'Terminal=true\n')
                f.write(f'Type=Application\n')
            
            os.chmod(shortcut_path, 0o755)
            print(f"✅ 已创建桌面快捷方式: {shortcut_path}")
            
    except Exception as e:
        print(f"⚠️ 创建桌面快捷方式失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="启动PBR材质识别系统")
    parser.add_argument("--create-shortcut", action="store_true", 
                       help="创建桌面快捷方式")
    parser.add_argument("--no-browser", action="store_true",
                       help="不自动打开浏览器")
    
    args = parser.parse_args()
    
    if args.create_shortcut:
        create_desktop_shortcut()
        return
    
    launcher = SystemLauncher()
    
    if args.no_browser:
        launcher.open_browser = lambda: None  # 禁用自动打开浏览器
    
    try:
        success = launcher.start()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户中断启动")
        launcher.stop()
    except Exception as e:
        print(f"启动失败: {e}")
        launcher.stop()
        sys.exit(1)


if __name__ == '__main__':
    main() 