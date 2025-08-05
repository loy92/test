#!/usr/bin/env python3
"""
提升PBR识别准确率的工具
包含数据收集、模型训练、参数调优等功能
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pbr_analyzer import create_pbr_analyzer
from PIL import Image

logger = logging.getLogger(__name__)


class PBRAccuracyImprover:
    """PBR识别准确率改进器"""
    
    def __init__(self, analyzer_type: str = 'enhanced'):
        self.analyzer = create_pbr_analyzer(use_enhanced=(analyzer_type == 'enhanced'))
        self.material_database = self._load_material_database()
        
    def _load_material_database(self) -> Dict:
        """加载材质数据库"""
        db_path = Path(__file__).parent.parent / "data" / "material_database.json"
        
        if db_path.exists():
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 创建默认材质数据库
            default_db = self._create_default_material_database()
            os.makedirs(db_path.parent, exist_ok=True)
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(default_db, f, indent=2, ensure_ascii=False)
            return default_db
    
    def _create_default_material_database(self) -> Dict:
        """创建默认材质数据库"""
        return {
            "materials": {
                "aluminum": {
                    "metallic": 0.95,
                    "roughness": 0.1,
                    "transparency": 0.0,
                    "normalStrength": 0.2,
                    "base_color": [180, 180, 180],
                    "keywords": ["铝", "aluminum", "metal", "silver"],
                    "description": "铝金属材质"
                },
                "steel": {
                    "metallic": 0.9,
                    "roughness": 0.2,
                    "transparency": 0.0,
                    "normalStrength": 0.3,
                    "base_color": [150, 150, 150],
                    "keywords": ["钢", "steel", "iron", "metal"],
                    "description": "钢铁材质"
                },
                "copper": {
                    "metallic": 0.95,
                    "roughness": 0.1,
                    "transparency": 0.0,
                    "normalStrength": 0.2,
                    "base_color": [184, 115, 51],
                    "keywords": ["铜", "copper", "bronze"],
                    "description": "铜金属材质"
                },
                "plastic": {
                    "metallic": 0.0,
                    "roughness": 0.5,
                    "transparency": 0.1,
                    "normalStrength": 0.2,
                    "base_color": [200, 200, 200],
                    "keywords": ["塑料", "plastic"],
                    "description": "塑料材质"
                },
                "glass": {
                    "metallic": 0.0,
                    "roughness": 0.0,
                    "transparency": 0.9,
                    "normalStrength": 0.1,
                    "base_color": [240, 240, 240],
                    "keywords": ["玻璃", "glass", "transparent"],
                    "description": "玻璃材质"
                },
                "wood": {
                    "metallic": 0.0,
                    "roughness": 0.8,
                    "transparency": 0.0,
                    "normalStrength": 0.6,
                    "base_color": [139, 69, 19],
                    "keywords": ["木头", "wood", "wooden"],
                    "description": "木质材质"
                },
                "fabric": {
                    "metallic": 0.0,
                    "roughness": 0.9,
                    "transparency": 0.0,
                    "normalStrength": 0.7,
                    "base_color": [100, 100, 100],
                    "keywords": ["布料", "fabric", "cloth", "textile"],
                    "description": "布料材质"
                },
                "ceramic": {
                    "metallic": 0.0,
                    "roughness": 0.1,
                    "transparency": 0.0,
                    "normalStrength": 0.2,
                    "base_color": [240, 240, 220],
                    "keywords": ["陶瓷", "ceramic", "porcelain"],
                    "description": "陶瓷材质"
                },
                "rubber": {
                    "metallic": 0.0,
                    "roughness": 0.8,
                    "transparency": 0.0,
                    "normalStrength": 0.4,
                    "base_color": [50, 50, 50],
                    "keywords": ["橡胶", "rubber"],
                    "description": "橡胶材质"
                },
                "gold": {
                    "metallic": 1.0,
                    "roughness": 0.1,
                    "transparency": 0.0,
                    "normalStrength": 0.1,
                    "base_color": [255, 215, 0],
                    "keywords": ["金", "gold", "golden"],
                    "description": "黄金材质"
                }
            },
            "accuracy_tips": {
                "lighting": "确保图像光照均匀，避免过强的阴影",
                "resolution": "使用512x512或更高分辨率的图像",
                "background": "使用简洁的背景，避免干扰",
                "angle": "正面或45度角拍摄效果最佳",
                "focus": "确保图像清晰，避免模糊"
            }
        }
    
    def analyze_with_reference(self, image_path: str, material_hint: str = None) -> Dict:
        """使用参考数据库进行分析"""
        # 加载图像
        image = Image.open(image_path)
        
        # 基础分析
        basic_result = self.analyzer.analyze_image(image)
        
        # 如果提供了材质提示，使用数据库修正
        if material_hint:
            corrected_result = self._apply_material_correction(basic_result, material_hint)
            return corrected_result
        else:
            # 自动匹配最接近的材质
            matched_material = self._match_material(basic_result)
            if matched_material:
                print(f"检测到可能的材质类型: {matched_material}")
                corrected_result = self._apply_material_correction(basic_result, matched_material)
                corrected_result['detected_material'] = matched_material
                return corrected_result
        
        return basic_result
    
    def _apply_material_correction(self, result: Dict, material_name: str) -> Dict:
        """应用材质数据库修正"""
        if material_name in self.material_database['materials']:
            reference = self.material_database['materials'][material_name]
            
            # 加权融合原始结果和参考值
            corrected = result.copy()
            weight = 0.3  # 参考权重
            
            for param in ['metallic', 'roughness', 'transparency', 'normalStrength']:
                if param in reference:
                    original = result[param]
                    reference_val = reference[param]
                    corrected[param] = original * (1 - weight) + reference_val * weight
            
            # 提高置信度
            corrected['confidence'] = min(result['confidence'] + 0.2, 1.0)
            corrected['correction_applied'] = material_name
            
            return corrected
        
        return result
    
    def _match_material(self, result: Dict) -> str:
        """自动匹配材质类型"""
        best_match = None
        min_distance = float('inf')
        
        for material_name, material_data in self.material_database['materials'].items():
            # 计算参数距离
            distance = 0
            for param in ['metallic', 'roughness', 'transparency', 'normalStrength']:
                if param in material_data:
                    distance += abs(result[param] - material_data[param])
            
            if distance < min_distance:
                min_distance = distance
                best_match = material_name
        
        # 如果距离足够小，返回匹配结果
        if min_distance < 1.0:  # 阈值可调
            return best_match
        
        return None
    
    def batch_analyze(self, image_directory: str, output_file: str = None) -> List[Dict]:
        """批量分析图像"""
        image_dir = Path(image_directory)
        results = []
        
        if not image_dir.exists():
            print(f"目录不存在: {image_directory}")
            return results
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for image_file in image_files:
            try:
                print(f"分析: {image_file.name}")
                result = self.analyze_with_reference(str(image_file))
                result['filename'] = image_file.name
                results.append(result)
                
                # 打印结果摘要
                print(f"  金属度: {result['metallic']:.2f}")
                print(f"  粗糙度: {result['roughness']:.2f}")
                print(f"  透明度: {result['transparency']:.2f}")
                print(f"  置信度: {result['confidence']:.2f}")
                if 'detected_material' in result:
                    print(f"  检测材质: {result['detected_material']}")
                print()
                
            except Exception as e:
                print(f"分析失败 {image_file.name}: {e}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output_file}")
        
        return results
    
    def calibrate_analyzer(self, calibration_data: List[Dict]) -> Dict:
        """校准分析器参数"""
        """
        calibration_data 格式:
        [
            {
                "image_path": "path/to/image.jpg",
                "ground_truth": {
                    "metallic": 0.8,
                    "roughness": 0.2,
                    "transparency": 0.0,
                    "normalStrength": 0.3
                }
            },
            ...
        ]
        """
        if not calibration_data:
            print("没有校准数据")
            return {}
        
        errors = {'metallic': [], 'roughness': [], 'transparency': [], 'normalStrength': []}
        
        print("开始校准分析...")
        for i, data in enumerate(calibration_data):
            try:
                image = Image.open(data['image_path'])
                predicted = self.analyzer.analyze_image(image)
                ground_truth = data['ground_truth']
                
                # 计算误差
                for param in errors.keys():
                    if param in ground_truth:
                        error = abs(predicted[param] - ground_truth[param])
                        errors[param].append(error)
                
                print(f"处理 {i+1}/{len(calibration_data)}: {data['image_path']}")
                
            except Exception as e:
                print(f"校准数据处理失败: {e}")
        
        # 计算统计信息
        stats = {}
        for param, error_list in errors.items():
            if error_list:
                stats[param] = {
                    'mean_error': np.mean(error_list),
                    'std_error': np.std(error_list),
                    'max_error': np.max(error_list),
                    'min_error': np.min(error_list)
                }
        
        print("\n校准结果:")
        for param, stat in stats.items():
            print(f"{param}:")
            print(f"  平均误差: {stat['mean_error']:.3f}")
            print(f"  标准差: {stat['std_error']:.3f}")
            print(f"  最大误差: {stat['max_error']:.3f}")
            print(f"  最小误差: {stat['min_error']:.3f}")
        
        return stats
    
    def get_accuracy_tips(self) -> List[str]:
        """获取提升准确率的建议"""
        tips = []
        for key, tip in self.material_database['accuracy_tips'].items():
            tips.append(f"{key}: {tip}")
        
        tips.extend([
            "多角度拍摄: 从不同角度拍摄同一材质可提高识别准确率",
            "标准化拍摄: 使用一致的拍摄条件和设置",
            "材质纯度: 避免混合材质，单一材质识别效果更好",
            "尺寸适中: 材质区域应占图像的主要部分",
            "颜色校准: 确保相机颜色准确，避免色偏"
        ])
        
        return tips


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PBR识别准确率改进工具")
    parser.add_argument('action', choices=['analyze', 'batch', 'calibrate', 'tips'],
                       help='执行的操作')
    parser.add_argument('--image', '-i', type=str, help='图像文件路径')
    parser.add_argument('--directory', '-d', type=str, help='图像目录路径')
    parser.add_argument('--output', '-o', type=str, help='输出文件路径')
    parser.add_argument('--material', '-m', type=str, help='材质提示')
    parser.add_argument('--calibration', '-c', type=str, help='校准数据文件')
    parser.add_argument('--analyzer', choices=['basic', 'enhanced'], default='enhanced',
                       help='使用的分析器类型')
    
    args = parser.parse_args()
    
    # 创建改进器
    improver = PBRAccuracyImprover(analyzer_type=args.analyzer)
    
    if args.action == 'analyze':
        if not args.image:
            print("请指定图像文件: --image IMAGE_PATH")
            return
        
        print(f"分析图像: {args.image}")
        result = improver.analyze_with_reference(args.image, args.material)
        
        print("\n分析结果:")
        print(f"金属度: {result['metallic']:.3f}")
        print(f"粗糙度: {result['roughness']:.3f}")
        print(f"透明度: {result['transparency']:.3f}")
        print(f"凹凸强度: {result['normalStrength']:.3f}")
        print(f"置信度: {result['confidence']:.3f}")
        
        if 'detected_material' in result:
            print(f"检测材质: {result['detected_material']}")
        if 'correction_applied' in result:
            print(f"应用修正: {result['correction_applied']}")
    
    elif args.action == 'batch':
        if not args.directory:
            print("请指定图像目录: --directory DIRECTORY_PATH")
            return
        
        output_file = args.output or 'batch_analysis_results.json'
        results = improver.batch_analyze(args.directory, output_file)
        print(f"批量分析完成，共处理 {len(results)} 个文件")
    
    elif args.action == 'calibrate':
        if not args.calibration:
            print("请指定校准数据文件: --calibration CALIBRATION_FILE")
            return
        
        try:
            with open(args.calibration, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            stats = improver.calibrate_analyzer(calibration_data)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2)
                print(f"校准结果已保存到: {args.output}")
        
        except Exception as e:
            print(f"校准失败: {e}")
    
    elif args.action == 'tips':
        print("提升PBR识别准确率的建议:")
        tips = improver.get_accuracy_tips()
        for i, tip in enumerate(tips, 1):
            print(f"{i}. {tip}")


if __name__ == '__main__':
    main() 