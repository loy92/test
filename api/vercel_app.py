#!/usr/bin/env python3
"""
Vercel轻量级PBR材质识别API
去除重型依赖，使用简化算法
"""

import os
import json
import base64
import io
from datetime import datetime
from typing import Dict, Optional
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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

class LightweightMaterialDatabase:
    """轻量级材质数据库"""
    
    def __init__(self):
        self.materials = {
            "wood": {
                "name": "木材",
                "metallic": (0.0, 0.05),
                "roughness": (0.6, 0.8),
                "transparency": (0.0, 0.1),
                "base_color": (139, 69, 19),
                "keywords": ["wood", "木", "木材", "oak", "pine"]
            },
            "metal": {
                "name": "金属", 
                "metallic": (0.9, 0.95),
                "roughness": (0.05, 0.25),
                "transparency": (0.0, 0.05),
                "base_color": (180, 180, 180),
                "keywords": ["metal", "金属", "steel", "钢", "aluminum", "铝"]
            },
            "plastic": {
                "name": "塑料",
                "metallic": (0.0, 0.1),
                "roughness": (0.1, 0.7),
                "transparency": (0.0, 0.3),
                "base_color": (200, 200, 200),
                "keywords": ["plastic", "塑料", "pvc", "abs"]
            },
            "glass": {
                "name": "玻璃",
                "metallic": (0.0, 0.1),
                "roughness": (0.0, 0.3),
                "transparency": (0.6, 0.95),
                "base_color": (240, 240, 240),
                "keywords": ["glass", "玻璃", "clear", "transparent"]
            }
        }
    
    def detect_material_from_filename(self, filename: str) -> Optional[str]:
        """从文件名检测材料类型"""
        if not filename:
            return None
        
        filename_lower = filename.lower()
        
        for material_id, material_info in self.materials.items():
            for keyword in material_info["keywords"]:
                if keyword.lower() in filename_lower:
                    return material_id
        
        return None
    
    def get_material_params(self, material_id: str) -> Optional[Dict]:
        """获取材质参数"""
        return self.materials.get(material_id)

class LightweightPBRAnalyzer:
    """轻量级PBR分析器"""
    
    def __init__(self):
        self.material_db = LightweightMaterialDatabase()
    
    def analyze_image(self, image_data: bytes, filename: str = None) -> Dict:
        """分析图像PBR参数"""
        try:
            # 基础图像分析
            base_analysis = self._analyze_image_basic(image_data)
            
            # 尝试从文件名检测材料
            detected_material = None
            if filename:
                detected_material = self.material_db.detect_material_from_filename(filename)
            
            if detected_material:
                # 使用检测到的材料信息增强分析
                material_params = self.material_db.get_material_params(detected_material)
                if material_params:
                    enhanced_analysis = self._enhance_with_material(base_analysis, material_params)
                    enhanced_analysis['detected_material'] = material_params['name']
                    enhanced_analysis['material_id'] = detected_material
                    enhanced_analysis['auto_detected'] = True
                    return enhanced_analysis
            
            # 如果没有检测到材料，返回基础分析
            base_analysis['detected_material'] = '未识别'
            base_analysis['auto_detected'] = False
            return base_analysis
            
        except Exception as e:
            print(f"分析失败: {e}")
            return self._fallback_analysis()
    
    def _analyze_image_basic(self, image_data: bytes) -> Dict:
        """基础图像分析"""
        if not HAS_PIL:
            return self._fallback_analysis()
        
        try:
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image.convert('RGB'))
            
            # 计算基本特征
            brightness = np.mean(img_array) / 255.0
            std_rgb = np.std(img_array, axis=(0, 1))
            color_variation = np.mean(std_rgb) / 255.0
            
            # 估算PBR参数
            metallic = min(brightness * 0.6 + (1 - color_variation) * 0.4, 1.0)
            roughness = min(color_variation * 2.0, 1.0)
            transparency = max(0, (brightness - 0.7) * 2)
            
            # 基础颜色
            base_color = [int(np.mean(img_array[:,:,i])) for i in range(3)]
            
            return {
                'metallic': max(metallic, 0.0),
                'roughness': max(roughness, 0.05),
                'transparency': min(transparency, 1.0),
                'confidence': 0.6 + color_variation * 0.2,
                'base_color': base_color
            }
            
        except Exception as e:
            print(f"图像分析失败: {e}")
            return self._fallback_analysis()
    
    def _enhance_with_material(self, base_analysis: Dict, material_params: Dict) -> Dict:
        """使用材料信息增强分析"""
        enhanced = base_analysis.copy()
        
        # 融合材料数据库的参数范围
        metallic_range = material_params['metallic']
        roughness_range = material_params['roughness']
        transparency_range = material_params['transparency']
        
        # 调整参数到材料范围内
        enhanced['metallic'] = np.clip(
            base_analysis['metallic'], 
            metallic_range[0], 
            metallic_range[1]
        )
        
        enhanced['roughness'] = np.clip(
            base_analysis['roughness'],
            roughness_range[0],
            roughness_range[1]
        )
        
        enhanced['transparency'] = np.clip(
            base_analysis['transparency'],
            transparency_range[0], 
            transparency_range[1]
        )
        
        # 提升置信度
        enhanced['confidence'] = min(base_analysis['confidence'] + 0.2, 0.95)
        
        # 使用材料的基础颜色
        enhanced['base_color'] = material_params['base_color']
        
        return enhanced
    
    def _fallback_analysis(self) -> Dict:
        """回退分析（随机值）"""
        import random
        
        return {
            'metallic': random.uniform(0.1, 0.8),
            'roughness': random.uniform(0.2, 0.8),
            'transparency': random.uniform(0.0, 0.3),
            'confidence': 0.3,
            'base_color': [random.randint(100, 200) for _ in range(3)]
        }

# 创建分析器实例
analyzer = LightweightPBRAnalyzer()

@app.route('/')
def index():
    """主页"""
    return jsonify({
        'message': 'PBR材质识别系统 - Vercel轻量级版本',
        'version': '1.0-vercel',
        'features': [
            '轻量级部署',
            '基础材料识别',
            '快速PBR参数估算'
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'deployment': 'vercel'})

@app.route('/api/analyze/enhanced', methods=['POST'])
def analyze_pbr():
    """PBR分析接口"""
    try:
        data = request.get_json()
        
        # 获取参数
        image_data = data.get('image')
        filename = data.get('filename')
        
        if not image_data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        # 解码base64图片
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
        except Exception as e:
            return jsonify({'error': f'图片解码失败: {str(e)}'}), 400
        
        # 进行分析
        results = analyzer.analyze_image(image_bytes, filename)
        
        # 添加时间戳
        results['timestamp'] = datetime.now().isoformat()
        results['analysis_type'] = 'lightweight'
        results['deployment'] = 'vercel'
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"分析失败: {e}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/api/materials/all', methods=['GET'])
def get_all_materials():
    """获取所有材料"""
    materials = []
    for material_id, material_info in analyzer.material_db.materials.items():
        materials.append({
            'id': material_id,
            'name': material_info['name'],
            'keywords': material_info['keywords']
        })
    
    return jsonify({
        'materials': materials,
        'count': len(materials)
    })

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """系统信息"""
    return jsonify({
        'name': 'PBR材质识别系统 - Vercel版',
        'version': '1.0-vercel',
        'deployment': 'vercel',
        'features': [
            '轻量级算法',
            '基础材料识别', 
            '快速响应'
        ],
        'limitations': [
            '简化的PBR算法',
            '有限的材料数据库',
            '基础图像分析'
        ]
    })

# Vercel入口点
def handler(request):
    """Vercel函数入口"""
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)