#!/usr/bin/env python3
"""
训练PBR-NeRF模型的脚本
使用方法：python train_pbr_model.py --config configs/default_config.json
"""

import argparse
import json
import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training.trainer import PBRTrainer
    from data.dataset import create_synthetic_pbr_data
except ImportError as e:
    print(f"警告: 无法导入训练模块 {e}")
    print("请确保已安装所有依赖: pip3 install -r requirements.txt")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练PBR-NeRF模型')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default_config.json',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        help='数据集目录（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='输出目录（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='训练轮数（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='使用的设备'
    )
    
    parser.add_argument(
        '--create_demo_data',
        action='store_true',
        help='创建演示数据集'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='从检查点恢复训练'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def create_demo_dataset():
    """创建演示数据集"""
    demo_data_dir = 'data/demo_pbr_dataset'
    print(f"正在创建演示数据集: {demo_data_dir}")
    
    create_synthetic_pbr_data(
        output_dir=demo_data_dir,
        num_images=50,  # 较小的数据集用于快速测试
        image_size=(256, 256)
    )
    
    print(f"演示数据集创建完成: {demo_data_dir}")
    return demo_data_dir


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("PBR-NeRF模型训练")
    print("=" * 60)
    
    # 加载配置
    config = load_config(args.config)
    print(f"加载配置文件: {args.config}")
    
    # 命令行参数覆盖配置文件
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # 创建演示数据集（如果需要）
    if args.create_demo_data:
        demo_data_dir = create_demo_dataset()
        config['data']['data_dir'] = demo_data_dir
    
    # 检查数据集是否存在
    data_dir = config['data']['data_dir']
    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录不存在: {data_dir}")
        print("请使用 --create_demo_data 创建演示数据集，或指定有效的数据集路径")
        return
    
    # 检查设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
        device = 'cpu'
    
    print(f"使用设备: {device}")
    print(f"数据集目录: {data_dir}")
    print(f"输出目录: {config['training']['output_dir']}")
    print(f"训练轮数: {config['training']['num_epochs']}")
    
    # 创建训练器
    trainer = PBRTrainer(config, device=device)
    
    # 从检查点恢复（如果指定）
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
            print(f"从检查点恢复训练: {args.resume}")
        else:
            print(f"警告: 检查点文件不存在: {args.resume}")
    
    # 开始训练
    try:
        print("\n开始训练...")
        trainer.train(config['training']['num_epochs'])
        print("\n训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("保存当前检查点...")
        trainer.save_checkpoint()
        print("检查点已保存")
        
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("保存当前检查点...")
        trainer.save_checkpoint()
        print("检查点已保存")
        raise


if __name__ == '__main__':
    main() 