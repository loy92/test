#!/usr/bin/env python3
"""
简化版PBR材质识别系统API服务器
无需复杂深度学习依赖，基于图像分析提供PBR参数预测
"""

import os
import json
import uuid
import base64
import io
from datetime import datetime
from typing import Dict
import numpy as np

# 尝试导入PIL，如果没有则使用替代方案
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

class SimplePBRAnalyzer:
    """基于图像统计的简化PBR分析器"""
    
    def analyze_image(self, image_data):
        """分析图像并预测PBR参数"""
        try:
            if HAS_PIL:
                return self._analyze_with_pil(image_data)
            else:
                return self._basic_analysis()
        except Exception as e:
            print(f"分析错误: {e}")
            return self._basic_analysis()
    
    def _analyze_with_pil(self, image_data):
        """使用PIL进行图像分析"""
        try:
            # 解码base64图像
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # 添加填充以确保base64解码正确
                missing_padding = len(image_data) % 4
                if missing_padding:
                    image_data += '=' * (4 - missing_padding)
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_data
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 计算基本统计信息
            brightness = np.mean(img_array) / 255.0
            std_rgb = np.std(img_array, axis=(0, 1))
            color_variation = np.mean(std_rgb) / 255.0
            
            # 计算颜色通道统计
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            
            # 计算反射特征
            reflection_score = self._calculate_reflection_score(img_array)
            
            # 基于统计特征预测PBR参数
            metallic = self._predict_metallic(brightness, color_variation, reflection_score)
            roughness = self._predict_roughness(color_variation, std_rgb)
            transparency = self._predict_transparency(brightness, img_array)
            
            # 计算置信度 - 基于图像质量和特征明显程度
            image_quality = min(img_array.shape[0] * img_array.shape[1] / 262144, 1.0)  # 512x512为基准
            feature_clarity = color_variation * 2 + reflection_score
            confidence = min(0.85, max(0.4, brightness * feature_clarity * image_quality))
            
            return {
                'metallic': float(metallic),
                'roughness': float(roughness), 
                'transparency': float(transparency),
                'normalStrength': float(min(color_variation * 1.5, 0.8)),
                'confidence': float(confidence),
                'base_color': [int(r_mean), int(g_mean), int(b_mean)],
                'analysis_method': 'PIL图像分析'
            }
            
        except Exception as e:
            print(f"PIL分析错误: {e}")
            return self._basic_analysis()
    
    def _calculate_reflection_score(self, img_array):
        """计算反射得分"""
        # 计算亮度分布
        brightness = np.mean(img_array, axis=2)
        
        # 检测高亮区域（可能的反射）
        bright_threshold = np.percentile(brightness, 90)
        bright_pixels = np.sum(brightness > bright_threshold)
        total_pixels = brightness.size
        
        reflection_ratio = bright_pixels / total_pixels
        return min(reflection_ratio * 3, 1.0)
    
    def _predict_metallic(self, brightness, variation, reflection):
        """预测金属度"""
        # 金属材质特征：高反射率、较低颜色变化、中高亮度
        if brightness > 0.7 and reflection > 0.3 and variation < 0.4:
            # 典型金属特征
            metallic = 0.6 + (brightness * 0.2) + (reflection * 0.15) + ((1 - variation) * 0.05)
        elif brightness < 0.3 or variation > 0.6:
            # 典型非金属特征
            metallic = max(0.1, brightness * 0.3)
        else:
            # 中等特征
            metallic = (brightness * 0.3 + reflection * 0.4 + (1 - variation) * 0.3)
        
        return min(max(metallic, 0.0), 1.0)
    
    def _predict_roughness(self, variation, std_rgb):
        """预测粗糙度"""
        # 纹理变化越大，粗糙度越高
        texture_score = min(variation * 1.5, 1.0)
        
        # 颜色通道一致性 - 不一致表示表面不平整
        color_consistency = np.std(std_rgb) / (np.mean(std_rgb) + 1e-6)
        color_roughness = min(color_consistency * 0.8, 0.8)
        
        # 综合粗糙度
        if texture_score > 0.6:
            # 高纹理变化 = 粗糙表面
            roughness = 0.7 + texture_score * 0.25 + color_roughness * 0.05
        elif texture_score < 0.2:
            # 低纹理变化 = 光滑表面
            roughness = texture_score * 0.4 + color_roughness * 0.1
        else:
            # 中等纹理
            roughness = texture_score * 0.6 + color_roughness * 0.4
        
        return min(max(roughness, 0.05), 0.95)
    
    def _predict_transparency(self, brightness, img_array):
        """预测透明度"""
        # 检测可能的透明效果
        if brightness > 0.8:
            # 很亮可能是透明材质
            transparency = (brightness - 0.8) * 5
        else:
            # 检测边缘透明度
            edge_brightness = self._calculate_edge_brightness(img_array)
            transparency = max(0, (edge_brightness - 0.6) * 2)
        
        return min(max(transparency, 0.0), 1.0)
    
    def _calculate_edge_brightness(self, img_array):
        """计算边缘亮度"""
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
        
        return edge_brightness
    
    def _basic_analysis(self):
        """基础分析（无PIL时的回退方案）"""
        # 返回常见材质的典型参数范围
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
            'confidence': 0.25,  # 较低置信度表示这是估算
            'base_color': [128, 128, 128],
            'analysis_method': f'模拟{material["name"]}材质',
            'note': '图像解码失败，使用材质类型估算'
        }

# 创建分析器实例
analyzer = SimplePBRAnalyzer()

@app.route('/')
def index():
    """主页"""
    return jsonify({
        'message': '简化版PBR材质识别系统API',
        'version': '1.0.0-simple',
        'status': 'running',
        'features': {
            'PIL': HAS_PIL,
            'numpy': True
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'type': 'simple_api'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_pbr():
    """分析PBR参数接口"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        image_data = data['image']
        settings = data.get('settings', {})
        
        # 分析图像
        results = analyzer.analyze_image(image_data)
        
        # 添加时间戳和设置信息
        results['timestamp'] = datetime.now().isoformat()
        results['settings'] = settings
        
        print(f"✅ PBR分析完成: metallic={results['metallic']:.3f}, "
              f"roughness={results['roughness']:.3f}, "
              f"confidence={results['confidence']:.3f}")
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """获取系统信息"""
    return jsonify({
        'type': 'simple_api',
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        'features': {
            'PIL': HAS_PIL,
            'numpy': True,
            'torch': False
        }
    })

if __name__ == '__main__':
    print("🚀 启动简化版PBR材质识别系统API...")
    print(f"📦 PIL支持: {'✅' if HAS_PIL else '❌'}")
    
    # 尝试多个端口
    ports = [5001, 5002, 5003, 5000]
    for port in ports:
        try:
            print(f"🔌 尝试在端口 {port} 启动服务器...")
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False
            )
            print(f"✅ 服务器成功在端口 {port} 启动")
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"❌ 端口 {port} 被占用，尝试下一个...")
                continue
            else:
                print(f"❌ 启动失败: {e}")
                continue
    else:
        print("❌ 无法找到可用端口")