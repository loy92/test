"""
API服务器模块

提供REST API接口用于PBR材质参数预测
"""

from .app import app, create_app, PBRPredictor

__all__ = [
    'app',
    'create_app',
    'PBRPredictor'
] 