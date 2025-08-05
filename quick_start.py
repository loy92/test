#!/usr/bin/env python3
"""
PBR材质识别系统 - 快速启动脚本
一键启动完整的PBR材质识别系统
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数 - 启动系统"""
    print("🚀 PBR材质识别系统 - 快速启动")
    print("=" * 50)
    print("基于Google Nerfies的智能材质参数识别")
    print("支持金属度、粗糙度、透明度、凹凸检测")
    print("=" * 50)
    
    try:
        # 导入启动器
        from scripts.start_system import SystemLauncher
        
        # 创建并启动系统
        launcher = SystemLauncher()
        launcher.start()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("\n请确保已安装所有依赖:")
        print("pip install -r requirements.txt")
        
    except KeyboardInterrupt:
        print("\n用户中断启动")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n请检查:")
        print("1. Python版本是否≥3.8")
        print("2. 是否已安装所有依赖")
        print("3. 端口5000和8080是否被占用")


if __name__ == '__main__':
    main() 