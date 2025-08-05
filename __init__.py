"""
PBR材质识别系统 - 基于Nerfies神经辐射场

本包提供了基于Google Nerfies扩展的PBR材质参数识别系统，
包括神经网络模型、训练工具、Web界面和API服务器。

主要模块：
- models: 神经网络模型定义
- data: 数据加载和处理
- training: 模型训练
- api: API服务器
- web: Web前端界面
"""

__version__ = "1.0.0"
__author__ = "PBR-Nerfies Team"
__email__ = "contact@pbr-nerfies.com"
__description__ = "基于Nerfies的PBR材质参数识别系统"

# 导入主要组件
try:
    from .models.pbr_nerf import PBRNeRF, PBRLoss, create_pbr_nerf_model
    from .training.trainer import PBRTrainer
    from .data.dataset import PBRDataset, PBRDataLoader
    
    __all__ = [
        'PBRNeRF',
        'PBRLoss', 
        'create_pbr_nerf_model',
        'PBRTrainer',
        'PBRDataset',
        'PBRDataLoader'
    ]
    
except ImportError:
    # 如果依赖缺失，提供友好的错误信息
    import warnings
    warnings.warn(
        "某些依赖包缺失，请运行 'pip install -r requirements.txt' 安装所需依赖",
        ImportWarning
    )
    __all__ = []


def get_version():
    """获取版本信息"""
    return __version__


def get_info():
    """获取包信息"""
    return {
        'name': 'pbr_nerfies_system',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'url': 'https://github.com/your-org/pbr-nerfies-system'
    } 