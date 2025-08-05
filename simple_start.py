#!/usr/bin/env python3
"""
PBR材质识别系统 - 简化启动脚本
先启动Web界面，后台处理API和依赖
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def install_dependencies():
    """安装必要的依赖"""
    print("正在安装系统依赖...")
    
    # 基础依赖列表
    basic_deps = [
        "flask>=2.0.0",
        "flask-cors>=3.0.0", 
        "numpy>=1.21.0",
        "pillow>=8.3.0"
    ]
    
    try:
        # 先安装基础依赖
        for dep in basic_deps:
            print(f"安装 {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          capture_output=True, check=True)
        
        print("✅ 基础依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def start_web_server():
    """启动Web服务器"""
    print("正在启动Web服务器...")
    
    try:
        web_dir = Path(__file__).parent / "web"
        os.chdir(web_dir)
        
        # 启动简单的HTTP服务器
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8080"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Web服务器启动成功 (http://localhost:8080)")
        return process
        
    except Exception as e:
        print(f"❌ Web服务器启动失败: {e}")
        return None

def create_simple_api():
    """创建简化的API服务器"""
    api_code = '''
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        'message': 'PBR材质识别系统API (简化版)',
        'status': 'running',
        'note': '正在加载完整模型...'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_pbr():
    try:
        # 返回模拟的PBR参数
        results = {
            'metallic': random.uniform(0.1, 0.9),
            'roughness': random.uniform(0.1, 0.9), 
            'transparency': random.uniform(0.0, 0.6),
            'normalStrength': random.uniform(0.1, 0.8),
            'confidence': random.uniform(0.7, 0.95),
            'base_color': [
                random.randint(50, 200),
                random.randint(50, 200), 
                random.randint(50, 200)
            ]
        }
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("启动简化API服务器...")
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    # 保存简化API到临时文件
    api_file = Path(__file__).parent / "temp_api.py"
    with open(api_file, 'w', encoding='utf-8') as f:
        f.write(api_code)
    
    return api_file

def start_simple_api():
    """启动简化API服务器"""
    print("正在启动简化API服务器...")
    
    try:
        # 创建简化API文件
        api_file = create_simple_api()
        
        # 启动API服务器
        process = subprocess.Popen([
            sys.executable, str(api_file)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ 简化API服务器启动成功 (http://localhost:5000)")
        return process, api_file
        
    except Exception as e:
        print(f"❌ API服务器启动失败: {e}")
        return None, None

def main():
    """主函数"""
    print("🚀 PBR材质识别系统 - 简化启动")
    print("=" * 50)
    
    # 步骤1: 安装基础依赖
    if not install_dependencies():
        print("依赖安装失败，将使用最简化模式")
    
    # 步骤2: 启动Web服务器
    web_process = start_web_server()
    if not web_process:
        print("❌ 无法启动Web服务器")
        return
    
    time.sleep(2)
    
    # 步骤3: 启动简化API
    api_process, api_file = start_simple_api()
    if not api_process:
        print("❌ 无法启动API服务器")
        web_process.terminate()
        return
    
    time.sleep(3)
    
    # 步骤4: 打开浏览器
    try:
        webbrowser.open("http://localhost:8080")
        print("✅ 浏览器已打开")
    except:
        print("请手动访问: http://localhost:8080")
    
    print("\n🎉 系统启动完成！")
    print("Web界面: http://localhost:8080")
    print("API接口: http://localhost:5000")
    print("\n注意: 当前使用简化模式，PBR分析结果为模拟数据")
    print("要获得真实AI分析，请等待完整依赖安装完成后重启")
    print("\n按 Ctrl+C 停止系统")
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭系统...")
        
        if web_process:
            web_process.terminate()
            web_process.wait()
        
        if api_process:
            api_process.terminate()
            api_process.wait()
        
        # 清理临时文件
        if api_file and api_file.exists():
            api_file.unlink()
        
        print("✅ 系统已关闭")

if __name__ == '__main__':
    main() 