"""
模型训练模块

包含训练器、验证和模型保存功能
"""

from .trainer import PBRTrainer, create_trainer_from_config

__all__ = [
    'PBRTrainer',
    'create_trainer_from_config'
] 