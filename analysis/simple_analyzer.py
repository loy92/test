"""
简化版PBR分析器
当完整版分析器无法导入时的备用方案
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SimplePBRAnalyzer:
    """简化版PBR分析器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        logger.info("简化PBR分析器已加载")
    
    def analyze_image(self, image: Image.Image) -> Dict[str, float]:
        """分析图像的PBR参数（简化版）"""
        try:
            # 转换为numpy数组
            img_array = np.array(image.convert('RGB'))
            
            # 基本图像统计
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            
            # 计算颜色特征
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            # 计算边缘强度
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 基于启发式规则估算PBR参数
            metallic = self._estimate_metallic(brightness, saturation, contrast)
            roughness = self._estimate_roughness(edge_density, contrast)
            transparency = self._estimate_transparency(brightness, saturation)
            normal_strength = self._estimate_normal_strength(edge_density)
            
            # 基础颜色
            base_color = [int(np.mean(img_array[:,:,i])) for i in range(3)]
            
            return {
                'metallic': float(np.clip(metallic, 0, 1)),
                'roughness': float(np.clip(roughness, 0, 1)),
                'transparency': float(np.clip(transparency, 0, 1)),
                'normalStrength': float(np.clip(normal_strength, 0, 1)),
                'confidence': 0.75,  # 较高置信度
                'base_color': base_color,
                'analysis_method': 'simplified_cv'
            }
            
        except Exception as e:
            logger.error(f"简化PBR分析失败: {e}")
            return self._get_fallback_result()
    
    def _estimate_metallic(self, brightness: float, saturation: float, contrast: float) -> float:
        """估算金属度（改进版）"""
        metallic_score = 0
        
        # 金属材质通常：高亮度、低饱和度、高对比度
        if brightness > 0.6:
            metallic_score += min((brightness - 0.6) / 0.4, 1) * 0.4
        
        if saturation < 0.3:
            metallic_score += min((0.3 - saturation) / 0.3, 1) * 0.4
        
        if contrast > 0.2:
            metallic_score += min(contrast / 0.5, 1) * 0.2
        
        return metallic_score
    
    def _estimate_roughness(self, edge_density: float, contrast: float) -> float:
        """估算粗糙度（改进版）"""
        roughness_score = 0
        
        # 粗糙表面通常：高边缘密度、高对比度变化
        roughness_score += min(edge_density * 4, 1) * 0.7
        roughness_score += min(contrast * 1.5, 1) * 0.3
        
        return roughness_score
    
    def _estimate_transparency(self, brightness: float, saturation: float) -> float:
        """估算透明度（改进版）"""
        transparency_score = 0
        
        # 透明材质通常：高亮度、极低饱和度
        if brightness > 0.85:
            transparency_score += min((brightness - 0.85) / 0.15, 1) * 0.6
        
        if saturation < 0.15:
            transparency_score += min((0.15 - saturation) / 0.15, 1) * 0.4
        
        return transparency_score
    
    def _estimate_normal_strength(self, edge_density: float) -> float:
        """估算法线强度（改进版）"""
        # 基于边缘密度，但限制最大值
        return min(edge_density * 3, 0.8)
    
    def _get_fallback_result(self) -> Dict[str, float]:
        """获取备用结果"""
        return {
            'metallic': 0.5,
            'roughness': 0.5,
            'transparency': 0.0,
            'normalStrength': 0.3,
            'confidence': 0.3,
            'base_color': [128, 128, 128],
            'analysis_method': 'fallback'
        }


def create_simple_analyzer(device: str = 'cpu') -> SimplePBRAnalyzer:
    """创建简化PBR分析器"""
    return SimplePBRAnalyzer(device=device) 