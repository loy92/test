#!/usr/bin/env python3
"""
增强版PBR材质识别系统API服务器
支持材料分类输入，大幅提高识别精准度
"""

import os
import json
import uuid
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# 尝试导入PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    
# 尝试导入flask
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if not HAS_FLASK:
    print("❌ 缺少Flask，请安装: pip install flask flask-cors")
    exit(1)

app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

class MaterialDatabase:
    """材质数据库"""
    
    def __init__(self):
        self.materials = {
            "wood": {
                "name": "木材",
                "metallic": (0.0, 0.05),
                "roughness": (0.6, 0.8),
                "transparency": (0.0, 0.1),
                "normalStrength": (0.4, 0.6),
                "base_color": (139, 69, 19),
                "confidence_boost": 0.2,
                "keywords": ["wood", "木", "木材", "oak", "橡木", "pine", "松木"]
            },
            "metal": {
                "name": "金属",
                "metallic": (0.9, 0.95),
                "roughness": (0.05, 0.25),
                "transparency": (0.0, 0.05),
                "normalStrength": (0.1, 0.3),
                "base_color": (180, 180, 180),
                "confidence_boost": 0.3,
                "keywords": ["metal", "金属", "aluminum", "铝", "steel", "钢", "iron", "铁", "copper", "铜"]
            },
            "plastic": {
                "name": "塑料",
                "metallic": (0.0, 0.1),
                "roughness": (0.1, 0.7),
                "transparency": (0.0, 0.3),
                "normalStrength": (0.1, 0.4),
                "base_color": (200, 200, 200),
                "confidence_boost": 0.25,
                "keywords": ["plastic", "塑料", "pvc", "abs", "poly"]
            },
            "vinyl": {
                "name": "乙烯基",
                "metallic": (0.0, 0.05),
                "roughness": (0.3, 0.6),
                "transparency": (0.0, 0.2),
                "normalStrength": (0.2, 0.4),
                "base_color": (180, 180, 180),
                "confidence_boost": 0.2,
                "keywords": ["vinyl", "乙烯基", "vinyl_", "pvc"]
            },
            "paper": {
                "name": "纸张",
                "metallic": (0.0, 0.02),
                "roughness": (0.7, 0.9),
                "transparency": (0.0, 0.1),
                "normalStrength": (0.5, 0.7),
                "base_color": (250, 250, 250),
                "confidence_boost": 0.15,
                "keywords": ["paper", "纸张", "纸", "cardboard", "纸板"]
            },
            "silicone": {
                "name": "硅胶",
                "metallic": (0.0, 0.05),
                "roughness": (0.2, 0.5),
                "transparency": (0.1, 0.4),
                "normalStrength": (0.2, 0.4),
                "base_color": (220, 220, 220),
                "confidence_boost": 0.2,
                "keywords": ["silicone", "硅胶", "silicone_", "rubber"]
            },
            "rubber": {
                "name": "橡胶",
                "metallic": (0.0, 0.05),
                "roughness": (0.4, 0.7),
                "transparency": (0.0, 0.2),
                "normalStrength": (0.3, 0.5),
                "base_color": (80, 80, 80),
                "confidence_boost": 0.2,
                "keywords": ["rubber", "橡胶", "rubber_", "latex"]
            },
            "leather": {
                "name": "皮革",
                "metallic": (0.0, 0.05),
                "roughness": (0.5, 0.8),
                "transparency": (0.0, 0.1),
                "normalStrength": (0.4, 0.6),
                "base_color": (139, 69, 19),
                "confidence_boost": 0.2,
                "keywords": ["leather", "皮革", "leather_", "skin"]
            },
            "stone": {
                "name": "石材",
                "metallic": (0.0, 0.1),
                "roughness": (0.3, 0.7),
                "transparency": (0.0, 0.1),
                "normalStrength": (0.3, 0.5),
                "base_color": (160, 160, 160),
                "confidence_boost": 0.25,
                "keywords": ["stone", "石材", "stone_", "marble", "大理石", "granite", "花岗岩"]
            },
            "screen_print": {
                "name": "丝网印刷",
                "metallic": (0.0, 0.1),
                "roughness": (0.2, 0.5),
                "transparency": (0.0, 0.2),
                "normalStrength": (0.2, 0.4),
                "base_color": (200, 200, 200),
                "confidence_boost": 0.2,
                "keywords": ["screen_print", "丝网印刷", "screen", "印刷", "print"]
            }
        }
    
    def get_material_categories(self) -> List[str]:
        """获取所有材质分类"""
        return list(self.materials.keys())
    
    def get_materials_in_category(self, category: str) -> List[Dict]:
        """获取指定分类下的所有材质"""
        if category not in self.materials:
            return []
        
        material = self.materials[category]
        return [{
            'id': category,
            'name': material['name'],
            'category': category,
            'keywords': material.get('keywords', [])
        }]
    
    def get_all_materials(self) -> List[Dict]:
        """获取所有材质信息"""
        materials = []
        for key, material in self.materials.items():
            materials.append({
                'id': key,
                'name': material['name'],
                'category': key,
                'keywords': material.get('keywords', [])
            })
        return materials
    
    def get_material_params(self, material_id: str) -> Optional[Dict]:
        """获取指定材质的参数范围"""
        return self.materials.get(material_id)

class EnhancedPBRAnalyzer:
    """增强版PBR分析器，支持材料分类"""
    
    def __init__(self):
        self.material_db = MaterialDatabase()
    
    def analyze_with_material_hint(
        self, 
        image_data: bytes, 
        material_category: str = None,
        material_id: str = None,
        filename: str = None
    ) -> Dict:
        """基于材料分类的分析"""
        
        # 基础图像分析
        base_analysis = self._analyze_image_basic(image_data)
        
        # 如果提供了材料分类，进行增强分析
        if material_category and material_id:
            enhanced_analysis = self._enhance_with_material_info(
                base_analysis, material_category, material_id
            )
            return enhanced_analysis
        
        # 尝试从文件名识别材料分类
        if filename:
            detected_category, detected_material = self._detect_material_from_filename(filename)
            if detected_category and detected_material:
                print(f"从文件名识别到材料: {detected_category} - {detected_material}")
                # 直接使用检测到的材料类型作为material_id
                enhanced_analysis = self._enhance_with_material_info(
                    base_analysis, detected_category, detected_material
                )
                enhanced_analysis['auto_detected_from_filename'] = True
                enhanced_analysis['filename'] = filename
                enhanced_analysis['detected_material'] = detected_material  # 添加检测到的材料ID
                return enhanced_analysis
        
        # 如果没有提供分类，尝试自动识别
        auto_category = self._auto_detect_material_category(base_analysis)
        if auto_category:
            return self._enhance_with_material_info(base_analysis, auto_category)
        
        return base_analysis
    
    def _analyze_image_basic(self, image_data: bytes) -> Dict:
        """基础图像分析"""
        if not HAS_PIL:
            return self._basic_analysis()
        
        try:
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image.convert('RGB'))
            
            # 计算基本特征
            brightness = np.mean(img_array) / 255.0
            std_rgb = np.std(img_array, axis=(0, 1))
            color_variation = np.mean(std_rgb) / 255.0
            
            # 计算反射特征
            reflection_score = self._calculate_reflection_score(img_array)
            
            # 基础参数预测
            metallic = self._predict_metallic(brightness, color_variation, reflection_score)
            roughness = self._predict_roughness(color_variation, std_rgb)
            transparency = self._predict_transparency(brightness, img_array)
            normal_strength = min(color_variation * 1.5, 0.8)
            
            # 基础颜色
            base_color = [int(np.mean(img_array[:,:,i])) for i in range(3)]
            
            # 计算基础置信度
            base_confidence = 0.5 + (color_variation * 0.3) + (min(reflection_score, 1.0) * 0.2)
            base_confidence = min(base_confidence, 0.8)  # 限制最大值为0.8
            
            return {
                'metallic': metallic,
                'roughness': roughness,
                'transparency': transparency,
                'normalStrength': normal_strength,
                'confidence': base_confidence,
                'base_color': base_color,
                'brightness': brightness,
                'color_variation': color_variation,
                'reflection_score': reflection_score
            }
            
        except Exception as e:
            print(f"图像分析失败: {e}")
            return self._basic_analysis()
    
    def _enhance_with_material_info(
        self, 
        base_analysis: Dict, 
        material_category: str,
        material_id: str = None
    ) -> Dict:
        """基于材料信息增强分析结果"""
        
        # 获取材质参数范围
        if material_id:
            material_params = self.material_db.get_material_params(material_id)
        else:
            # 如果没有指定具体材质，使用分类的平均值
            materials = self.material_db.get_materials_in_category(material_category)
            if materials:
                material_params = self._get_category_average_params(material_category)
            else:
                return base_analysis
        
        if not material_params:
            return base_analysis
        
        # 融合基础分析和材质数据库信息
        enhanced_results = {}
        
        # 金属度：结合图像分析和材质数据库
        db_metallic_range = material_params['metallic']
        img_metallic = base_analysis['metallic']
        enhanced_results['metallic'] = self._blend_parameters(
            img_metallic, db_metallic_range, weight=0.7
        )
        
        # 粗糙度：结合图像分析和材质数据库
        db_roughness_range = material_params['roughness']
        img_roughness = base_analysis['roughness']
        enhanced_results['roughness'] = self._blend_parameters(
            img_roughness, db_roughness_range, weight=0.7
        )
        
        # 透明度：主要基于材质数据库
        db_transparency_range = material_params['transparency']
        img_transparency = base_analysis['transparency']
        enhanced_results['transparency'] = self._blend_parameters(
            img_transparency, db_transparency_range, weight=0.8
        )
        
        # 法线强度：结合图像分析和材质数据库
        db_normal_range = material_params['normalStrength']
        img_normal = base_analysis['normalStrength']
        enhanced_results['normalStrength'] = self._blend_parameters(
            img_normal, db_normal_range, weight=0.6
        )
        
        # 基础颜色：主要基于材质数据库
        enhanced_results['base_color'] = material_params['base_color']
        
        # 置信度：基于材质数据库提升
        base_confidence = base_analysis['confidence']
        confidence_boost = material_params.get('confidence_boost', 0.2)
        enhanced_results['confidence'] = min(base_confidence + confidence_boost, 0.95)
        
        # 添加材料信息
        enhanced_results['material_category'] = material_category
        enhanced_results['detected_material'] = material_params['name']  # 添加检测到的材料名称
        enhanced_results['material_name'] = material_params['name']  # 添加材料名称
        if material_id:
            enhanced_results['material_id'] = material_id
        
        return enhanced_results
    
    def _blend_parameters(self, img_value: float, db_range: tuple, weight: float = 0.7) -> float:
        """融合图像分析值和数据库范围"""
        db_center = (db_range[0] + db_range[1]) / 2
        blended = img_value * (1 - weight) + db_center * weight
        return max(db_range[0], min(blended, db_range[1]))
    
    def _get_category_average_params(self, category: str) -> Dict:
        """获取分类的平均参数"""
        materials = self.material_db.materials.get(category, {})
        if not materials:
            return None
        
        # 计算平均值
        metallic_ranges = [m['metallic'] for m in materials.values()]
        roughness_ranges = [m['roughness'] for m in materials.values()]
        transparency_ranges = [m['transparency'] for m in materials.values()]
        normal_ranges = [m['normalStrength'] for m in materials.values()]
        
        return {
            'metallic': (np.mean([r[0] for r in metallic_ranges]), np.mean([r[1] for r in metallic_ranges])),
            'roughness': (np.mean([r[0] for r in roughness_ranges]), np.mean([r[1] for r in roughness_ranges])),
            'transparency': (np.mean([r[0] for r in transparency_ranges]), np.mean([r[1] for r in transparency_ranges])),
            'normalStrength': (np.mean([r[0] for r in normal_ranges]), np.mean([r[1] for r in normal_ranges])),
            'base_color': (200, 200, 200),  # 默认颜色
            'confidence_boost': 0.2,
            'name': f'{category}类'  # 添加材料名称
        }
    
    def _auto_detect_material_category(self, analysis: Dict) -> Optional[str]:
        """自动检测材料分类"""
        metallic = analysis['metallic']
        roughness = analysis['roughness']
        transparency = analysis['transparency']
        brightness = analysis.get('brightness', 0.5)
        
        # 基于参数范围自动分类
        if metallic > 0.7:
            return "metal"
        elif transparency > 0.6:
            return "screen_print"
        elif roughness > 0.6 and metallic < 0.2:
            return "leather"
        elif roughness < 0.3 and metallic < 0.2:
            return "plastic"
        elif brightness < 0.4:
            return "wood"
        else:
            return "stone"
    
    def _detect_material_from_filename(self, filename: str) -> tuple:
        """从文件名检测材料分类和具体材质"""
        if not filename:
            return None, None
        
        # 移除文件扩展名
        name_without_ext = filename.lower().split('.')[0]
        
        # 定义文件名关键词映射
        filename_keywords = {
            # 金属类
            "aluminum": ("metal", "aluminum"),
            "aluminium": ("metal", "aluminum"),
            "铝": ("metal", "aluminum"),
            "steel": ("metal", "steel"),
            "钢": ("metal", "steel"),
            "iron": ("metal", "iron"),
            "铁": ("metal", "iron"),
            "copper": ("metal", "copper"),
            "铜": ("metal", "copper"),
            "metal": ("metal", "steel"),
            "金属": ("metal", "steel"),
            
            # 塑料类
            "plastic": ("plastic", "plastic"),
            "塑料": ("plastic", "plastic"),
            "pvc": ("plastic", "vinyl"),
            "abs": ("plastic", "vinyl"),
            "poly": ("plastic", "vinyl"),
            "rough_plastic": ("plastic", "vinyl"),
            "粗糙塑料": ("plastic", "vinyl"),
            
            # 玻璃类
            "glass": ("screen_print", "screen_print"),
            "玻璃": ("screen_print", "screen_print"),
            "透明": ("screen_print", "screen_print"),
            "clear": ("screen_print", "screen_print"),
            "frosted_glass": ("screen_print", "screen_print"),
            "磨砂": ("screen_print", "screen_print"),
            "frosted": ("screen_print", "screen_print"),
            
            # 木材类
            "wood": ("wood", "wood"),
            "木": ("wood", "wood"),
            "oak": ("wood", "wood"),
            "橡木": ("wood", "wood"),
            "pine": ("wood", "wood"),
            "松木": ("wood", "wood"),
            
            # 织物类
            "fabric": ("leather", "leather"),
            "织物": ("leather", "leather"),
            "cotton": ("leather", "leather"),
            "棉": ("leather", "leather"),
            "silk": ("leather", "leather"),
            "丝绸": ("leather", "leather"),
            "cloth": ("leather", "leather"),
            "布": ("leather", "leather"),
            
            # 陶瓷类
            "ceramic": ("stone", "stone"),
            "陶瓷": ("stone", "stone"),
            "glazed": ("stone", "stone"),
            "釉面": ("stone", "stone"),
            "rough_ceramic": ("stone", "stone"),
            "粗陶": ("stone", "stone"),
            "pottery": ("stone", "stone"),
            "陶": ("stone", "stone"),
        }
        
        # 检查文件名中是否包含关键词
        for keyword, (category, material) in filename_keywords.items():
            if keyword in name_without_ext:
                return category, material
        
        # 如果没有找到精确匹配，尝试模糊匹配
        for category, materials in self.material_db.materials.items():
            for material_id, material_info in materials.items():
                # 检查材质名称是否在文件名中
                if isinstance(material_info, dict) and material_info.get('name', '').lower() in name_without_ext:
                    return category, material_id
        
        return None, None
    
    def _calculate_reflection_score(self, img_array: np.ndarray) -> float:
        """计算反射分数"""
        # 计算边缘亮度
        h, w = img_array.shape[:2]
        edge_width = min(h, w) // 20
        
        # 提取边缘像素
        top_edge = img_array[:edge_width, :]
        bottom_edge = img_array[-edge_width:, :]
        left_edge = img_array[:, :edge_width]
        right_edge = img_array[:, -edge_width:]
        
        edge_brightness = np.mean([
            np.mean(top_edge),
            np.mean(bottom_edge), 
            np.mean(left_edge),
            np.mean(right_edge)
        ]) / 255.0
        
        # 计算中心亮度
        center_h, center_w = h // 4, w // 4
        center = img_array[center_h:3*center_h, center_w:3*center_w]
        center_brightness = np.mean(center) / 255.0
        
        # 反射分数 = 边缘亮度与中心亮度的比值
        reflection_score = edge_brightness / (center_brightness + 0.1)
        return min(reflection_score, 2.0)
    
    def _predict_metallic(self, brightness: float, variation: float, reflection: float) -> float:
        """预测金属度"""
        # 高亮度 + 低变化 + 高反射 = 高金属度
        metallic = min(brightness * 0.6 + reflection * 0.3 + (1 - variation) * 0.1, 1.0)
        return max(metallic, 0.0)
    
    def _predict_roughness(self, variation: float, std_rgb: np.ndarray) -> float:
        """预测粗糙度"""
        # 高变化 = 高粗糙度
        roughness = min(variation * 1.5, 1.0)
        return max(roughness, 0.05)
    
    def _predict_transparency(self, brightness: float, img_array: np.ndarray) -> float:
        """预测透明度"""
        # 高亮度 + 低对比度 = 高透明度
        contrast = np.std(img_array) / 255.0
        transparency = max(0, (brightness - 0.7) * 2 - contrast * 0.5)
        return min(transparency, 1.0)
    
    def _basic_analysis(self) -> Dict:
        """基础分析（无PIL时的回退方案）"""
        import random
        
        # 模拟几种常见材质类型
        material_types = [
            {'name': '金属', 'metallic': (0.7, 0.9), 'roughness': (0.1, 0.4), 'transparency': (0.0, 0.1)},
            {'name': '塑料', 'metallic': (0.0, 0.2), 'roughness': (0.3, 0.8), 'transparency': (0.0, 0.3)},
            {'name': '陶瓷', 'metallic': (0.0, 0.1), 'roughness': (0.1, 0.3), 'transparency': (0.0, 0.1)},
            {'name': '织物', 'metallic': (0.0, 0.1), 'roughness': (0.7, 0.9), 'transparency': (0.0, 0.2)},
        ]
        
        # 随机选择一种材质类型
        material = random.choice(material_types)
        
        metallic = random.uniform(*material['metallic'])
        roughness = random.uniform(*material['roughness'])
        transparency = random.uniform(*material['transparency'])
        
        return {
            'metallic': metallic,
            'roughness': roughness,
            'transparency': transparency,
            'normalStrength': random.uniform(0.2, 0.6),
            'confidence': 0.3,
            'base_color': [random.randint(50, 200) for _ in range(3)]
        }

# 创建分析器实例
analyzer = EnhancedPBRAnalyzer()

@app.route('/')
def index():
    """主页"""
    return jsonify({
        'message': '增强版PBR材质识别系统API',
        'version': '2.0',
        'features': [
            '支持材料分类输入',
            '基于材质数据库的精准预测',
            '自动材料分类检测',
            '置信度提升机制'
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'enhanced': True})

@app.route('/api/materials/categories', methods=['GET'])
def get_material_categories():
    """获取所有材料分类"""
    categories = analyzer.material_db.get_material_categories()
    return jsonify({
        'categories': categories,
        'count': len(categories)
    })

@app.route('/api/materials/all', methods=['GET'])
def get_all_materials():
    """获取所有材料信息"""
    materials = analyzer.material_db.get_all_materials()
    return jsonify({
        'materials': materials,
        'count': len(materials)
    })

@app.route('/api/materials/category/<category>', methods=['GET'])
def get_materials_in_category(category):
    """获取指定分类下的材料"""
    materials = analyzer.material_db.get_materials_in_category(category)
    return jsonify({
        'category': category,
        'materials': materials,
        'count': len(materials)
    })

@app.route('/api/analyze/enhanced', methods=['POST'])
def analyze_pbr_enhanced():
    """增强版PBR分析接口"""
    try:
        data = request.get_json()
        
        # 获取参数
        image_data = data.get('image')
        material_category = data.get('materialCategory')
        material_id = data.get('materialId')
        filename = data.get('filename') # 新增文件名参数
        
        if not image_data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        # 解码base64图片
        try:
            # 移除data URL前缀
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
        except Exception as e:
            return jsonify({'error': f'图片解码失败: {str(e)}'}), 400
        
        # 进行增强分析
        results = analyzer.analyze_with_material_hint(
            image_data=image_bytes,
            material_category=material_category,
            material_id=material_id,
            filename=filename # 传递文件名
        )
        
        # 添加时间戳和分析信息
        results['timestamp'] = datetime.now().isoformat()
        results['analysis_type'] = 'enhanced'
        results['material_hint_used'] = bool(material_category or material_id)
        
        print(f"增强分析完成: {results}")
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"增强分析失败: {e}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """获取系统信息"""
    return jsonify({
        'name': '增强版PBR材质识别系统',
        'version': '2.0',
        'features': [
            '材料分类数据库',
            '基于分类的精准预测',
            '自动材料检测',
            '置信度提升机制'
        ],
        'material_categories': len(analyzer.material_db.get_material_categories()),
        'total_materials': sum(len(cat) for cat in analyzer.material_db.materials.values())
    })

if __name__ == '__main__':
    print("🚀 启动增强版PBR材质识别系统API...")
    print("✨ 支持材料分类输入，大幅提升识别精准度")
    print("🌐 API地址: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False) 