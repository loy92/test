"""
PBR材质识别系统API服务器
提供图片上传、PBR参数预测、模型管理等接口
"""

import os
import json
import uuid
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
from typing import Dict, Optional, Tuple
import io
import base64

# 使用相对导入和错误处理
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.pbr_nerf import create_pbr_nerf_model
    from analysis.pbr_analyzer import create_pbr_analyzer
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False
    def create_pbr_nerf_model(config):
        return None
    def create_pbr_analyzer(**kwargs):
        return None

try:
    from analysis.optimized_pbr_analyzer import create_optimized_analyzer
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    def create_optimized_analyzer(**kwargs):
        return None

try:
    from analysis.simple_analyzer import create_simple_analyzer
    HAS_SIMPLE = True
except ImportError:
    HAS_SIMPLE = False
    def create_simple_analyzer(**kwargs):
        return None

print(f"分析器状态: 优化={HAS_OPTIMIZED}, 基础={HAS_ANALYSIS}, 简化={HAS_SIMPLE}")


app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件大小
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型实例
pbr_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PBRPredictor:
    """PBR参数预测器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.analyzer = None
        
        # 依次尝试不同等级的PBR分析器
        self.analyzer = None
        
        # 1. 优先尝试优化分析器
        if HAS_OPTIMIZED:
            try:
                self.analyzer = create_optimized_analyzer(device=str(self.device))
                logger.info("✅ 已加载优化的PBR分析器")
            except Exception as e:
                logger.error(f"❌ 优化分析器加载失败: {e}")
        
        # 2. 回退到基础分析器
        if self.analyzer is None and HAS_ANALYSIS:
            try:
                self.analyzer = create_pbr_analyzer(use_enhanced=True, device=str(self.device))
                logger.info("✅ 已加载基础PBR分析器")
            except Exception as e:
                logger.error(f"❌ 基础分析器加载失败: {e}")
        
        # 3. 最终回退到简化分析器
        if self.analyzer is None and HAS_SIMPLE:
            try:
                self.analyzer = create_simple_analyzer(device=str(self.device))
                logger.info("✅ 已加载简化PBR分析器")
            except Exception as e:
                logger.error(f"❌ 简化分析器加载失败: {e}")
        
        if self.analyzer is None:
            logger.warning("⚠️ 无法加载任何PBR分析器，将使用内置简化分析")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 使用默认配置创建模型
            self.create_default_model()
    
    def create_default_model(self):
        """创建默认模型配置"""
        config = {
            'pos_encoding_freqs': 10,
            'dir_encoding_freqs': 4,
            'density_layers': 8,
            'density_hidden_dim': 256,
            'pbr_layers': 4,
            'pbr_hidden_dim': 128,
            'skip_connections': [4],
            'use_viewdirs': True,
            'deformation_layers': 6,
            'deformation_hidden_dim': 128
        }
        
        self.model = create_pbr_nerf_model(config)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"创建默认PBR模型，使用设备: {self.device}")
        else:
            logger.warning("无法创建PBR模型，将使用简化分析")
    
    def load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'config' in checkpoint:
                config = checkpoint['config']['model']
                self.model = create_pbr_nerf_model(config)
            else:
                self.create_default_model()
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"成功加载模型: {model_path}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.create_default_model()
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """预处理输入图像"""
        # 调整大小
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组并归一化
        image_array = np.array(image) / 255.0
        
        return image_array
    
    def generate_camera_rays(self, image_size: Tuple[int, int], num_rays: int = 1024) -> Dict:
        """生成相机光线用于渲染"""
        height, width = image_size
        
        # 简化的相机设置
        focal_length = max(width, height) * 0.8
        cx, cy = width / 2, height / 2
        
        # 随机采样像素
        coords = np.random.randint(0, min(width, height), size=(num_rays, 2))
        
        # 生成光线方向
        directions = np.zeros((num_rays, 3))
        directions[:, 0] = (coords[:, 0] - cx) / focal_length
        directions[:, 1] = -(coords[:, 1] - cy) / focal_length
        directions[:, 2] = -1.0
        
        # 归一化方向
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        # 光线原点（相机位置）
        origins = np.zeros((num_rays, 3))
        origins[:, 2] = 5.0  # 相机距离物体5个单位
        
        return {
            'rays_o': torch.from_numpy(origins).float().to(self.device),
            'rays_d': torch.from_numpy(directions).float().to(self.device),
            'coords': coords
        }
    
    def predict_pbr_parameters(
        self, 
        image: Image.Image, 
        num_samples: int = 64,
        enable_deformation: bool = True
    ) -> Dict:
        """预测PBR参数"""
        try:
            # 优先使用优化的PBR分析器
            if self.analyzer is not None:
                analyzer_type = getattr(self.analyzer, '__class__', type(self.analyzer)).__name__
                logger.info(f"使用{analyzer_type}分析器")
                results = self.analyzer.analyze_image(image)
                logger.info(f"分析结果: metallic={results.get('metallic', 0):.3f}, "
                          f"roughness={results.get('roughness', 0):.3f}, "
                          f"confidence={results.get('confidence', 0):.3f}")
                return results
            
            # 回退到NeRF模型（如果可用）
            if self.model is not None:
                logger.info("使用NeRF模型分析")
                return self._nerf_analysis(image, num_samples, enable_deformation)
            
            # 最终回退到基于图像特征的简单分析
            logger.info("使用简化图像分析")
            return self._simple_image_analysis(image)
            
        except Exception as e:
            logger.error(f"PBR参数预测失败: {e}")
            # 返回默认值
            return {
                'metallic': 0.5,
                'roughness': 0.5,
                'transparency': 0.0,
                'normalStrength': 0.3,
                'confidence': 0.1,
                'base_color': [128, 128, 128]
            }
    
    def _simple_image_analysis(self, image: Image.Image) -> Dict:
        """简化的图像分析（无需复杂依赖）"""
        try:
            import numpy as np
            
            # 转换为numpy数组
            img_array = np.array(image.convert('RGB'))
            
            # 计算基本统计信息
            brightness = np.mean(img_array) / 255.0
            
            # 计算颜色变化
            std_rgb = np.std(img_array, axis=(0, 1))
            color_variation = np.mean(std_rgb) / 255.0
            
            # 基于亮度和变化估算参数
            metallic = min(brightness * 0.8, 1.0) if brightness > 0.6 else 0.2
            roughness = min(color_variation * 2.0, 1.0)
            transparency = max(0, (brightness - 0.8) * 5) if brightness > 0.8 else 0.0
            normal_strength = min(color_variation * 1.5, 0.8)
            
            # 基础颜色
            base_color = [int(np.mean(img_array[:,:,i])) for i in range(3)]
            
            return {
                'metallic': metallic,
                'roughness': roughness, 
                'transparency': transparency,
                'normalStrength': normal_strength,
                'confidence': 0.6,
                'base_color': base_color
            }
            
        except Exception as e:
            logger.error(f"简化分析失败: {e}")
            return {
                'metallic': 0.5,
                'roughness': 0.5,
                'transparency': 0.0,
                'normalStrength': 0.3,
                'confidence': 0.1,
                'base_color': [128, 128, 128]
            }
    
    def _nerf_analysis(self, image: Image.Image, num_samples: int, enable_deformation: bool) -> Dict:
        """基于NeRF的分析（原始方法）"""
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)
            height, width = processed_image.shape[:2]
            
            # 生成相机光线
            rays_data = self.generate_camera_rays((height, width), num_rays=1024)
            
            # 添加变形编码（如果启用）
            deformation_code = None
            if enable_deformation:
                deformation_code = torch.zeros(rays_data['rays_o'].shape[0], 3).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                # 计算采样点
                near, far = 0.1, 10.0
                t_vals = torch.linspace(near, far, num_samples, device=self.device)
                t_vals = t_vals.expand(rays_data['rays_o'].shape[0], num_samples)
                
                # 计算3D点
                pts = rays_data['rays_o'][:, None, :] + rays_data['rays_d'][:, None, :] * t_vals[:, :, None]
                pts_flat = pts.reshape(-1, 3)
                
                # 扩展光线方向
                dirs_flat = rays_data['rays_d'][:, None, :].expand(-1, num_samples, -1).reshape(-1, 3)
                
                # 扩展变形编码
                if deformation_code is not None:
                    deform_flat = deformation_code[:, None, :].expand(-1, num_samples, -1).reshape(-1, 3)
                else:
                    deform_flat = None
                
                # 模型预测
                outputs = self.model(pts_flat, dirs_flat, deform_flat)
                
                # 重塑输出
                N, S = rays_data['rays_o'].shape[0], num_samples
                for key in outputs:
                    if outputs[key].dim() == 2:
                        outputs[key] = outputs[key].reshape(N, S, -1)
                    else:
                        outputs[key] = outputs[key].reshape(N, S)
                
                # 体积渲染
                rendered = self._volume_render(outputs, t_vals)
            
            # 计算平均PBR参数
            pbr_params = self._aggregate_pbr_parameters(rendered)
            
            # 添加置信度评估
            pbr_params['confidence'] = self._estimate_confidence(rendered)
            
            # 估算基础颜色
            pbr_params['base_color'] = self._estimate_base_color(processed_image)
            
            return pbr_params
            
        except Exception as e:
            logger.error(f"NeRF分析失败: {e}")
            return self._simple_image_analysis(image)
    
    def _volume_render(self, outputs: Dict, t_vals: torch.Tensor) -> Dict:
        """体积渲染"""
        density = outputs['density']
        
        # 计算距离
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # 计算透明度和权重
        alpha = 1.0 - torch.exp(-density * dists)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        weights = alpha * transmittance
        
        # 渲染各项参数
        result = {'weights': weights}
        
        for key in ['rgb', 'metallic', 'roughness', 'transparency', 'normal']:
            if key in outputs:
                if outputs[key].dim() == 3:  # [N, S, C]
                    result[key] = torch.sum(weights[..., None] * outputs[key], dim=-2)
                else:  # [N, S]
                    result[key] = torch.sum(weights * outputs[key], dim=-1)
        
        return result
    
    def _aggregate_pbr_parameters(self, rendered: Dict) -> Dict:
        """聚合PBR参数"""
        params = {}
        
        # 计算每个参数的平均值
        if 'metallic' in rendered:
            params['metallic'] = float(torch.mean(rendered['metallic']).cpu())
        else:
            params['metallic'] = 0.5
            
        if 'roughness' in rendered:
            params['roughness'] = float(torch.mean(rendered['roughness']).cpu())
        else:
            params['roughness'] = 0.5
            
        if 'transparency' in rendered:
            params['transparency'] = float(torch.mean(rendered['transparency']).cpu())
        else:
            params['transparency'] = 0.0
            
        if 'normal' in rendered:
            normal_strength = float(torch.mean(torch.norm(rendered['normal'], dim=-1)).cpu())
            params['normalStrength'] = min(normal_strength, 1.0)
        else:
            params['normalStrength'] = 0.3
        
        return params
    
    def _estimate_confidence(self, rendered: Dict) -> float:
        """估算预测置信度"""
        # 基于权重分布和参数方差来估算置信度
        if 'weights' in rendered:
            weight_entropy = -torch.sum(rendered['weights'] * torch.log(rendered['weights'] + 1e-10), dim=-1)
            confidence = 1.0 / (1.0 + float(torch.mean(weight_entropy).cpu()))
        else:
            confidence = 0.5
        
        return max(0.1, min(1.0, confidence))
    
    def _estimate_base_color(self, image: np.ndarray) -> list:
        """估算基础颜色"""
        # 计算图像平均颜色
        avg_color = np.mean(image.reshape(-1, 3), axis=0)
        return [int(c * 255) for c in avg_color]


# 初始化预测器
predictor = PBRPredictor()


@app.route('/')
def index():
    """主页"""
    return jsonify({
        'message': 'PBR材质识别系统API',
        'version': '1.0.0',
        'status': 'running',
        'device': str(device)
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """上传图片接口"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        logger.info(f"图片上传成功: {unique_filename}")
        
        return jsonify({
            'message': '图片上传成功',
            'filename': unique_filename,
            'filepath': filepath
        })
        
    except Exception as e:
        logger.error(f"图片上传失败: {e}")
        return jsonify({'error': '图片上传失败'}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_pbr():
    """分析PBR参数接口"""
    try:
        data = request.get_json()
        
        # 获取参数
        image_data = data.get('image')
        settings = data.get('settings', {})
        
        if not image_data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        # 解码base64图片
        try:
            # 移除data URL前缀
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            return jsonify({'error': f'图片解码失败: {str(e)}'}), 400
        
        # 预测PBR参数
        results = predictor.predict_pbr_parameters(
            image=image,
            num_samples=settings.get('numSamples', 64),
            enable_deformation=settings.get('enableDeformation', True)
        )
        
        # 添加时间戳
        results['timestamp'] = datetime.now().isoformat()
        results['settings'] = settings
        
        logger.info(f"PBR分析完成: {results}")
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"PBR分析失败: {e}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    models_dir = 'models'
    models = []
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.pth'):
                model_path = os.path.join(models_dir, filename)
                stat = os.stat(model_path)
                models.append({
                    'filename': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    return jsonify({'models': models})


@app.route('/api/models/load', methods=['POST'])
def load_model():
    """加载指定模型"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': '缺少模型名称'}), 400
        
        model_path = os.path.join('models', model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 404
        
        # 加载新模型
        global predictor
        predictor = PBRPredictor(model_path=model_path)
        
        return jsonify({
            'message': f'成功加载模型: {model_name}',
            'model_name': model_name
        })
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """获取系统信息"""
    return jsonify({
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'python_version': os.sys.version,
        'torch_version': torch.__version__,
        'model_loaded': predictor.model is not None
    })


@app.route('/api/debug', methods=['POST'])
def debug_analysis():
    """调试PBR分析"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        # 解码图片
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': f'图片解码失败: {str(e)}'}), 400
        
        debug_info = {}
        
        # 获取基础分析结果
        if predictor.analyzer and hasattr(predictor.analyzer, 'debug_features'):
            try:
                debug_info['features'] = predictor.analyzer.debug_features(image)
            except Exception as e:
                debug_info['features_error'] = str(e)
        
        # 获取分析结果
        pbr_result = predictor.predict_pbr_parameters(image)
        debug_info['pbr_result'] = pbr_result
        
        # 添加建议
        suggestions = []
        
        if pbr_result.get('confidence', 0) < 0.6:
            suggestions.append("置信度较低，建议:")
            suggestions.append("1. 检查图像清晰度和光照条件")
            suggestions.append("2. 确保材质占图像主要部分")
            suggestions.append("3. 避免复杂背景")
        
        if pbr_result.get('metallic', 0) > 0.8 and pbr_result.get('transparency', 0) > 0.3:
            suggestions.append("参数异常：高金属度材质不应该有高透明度")
            suggestions.append("可能原因：图像反光过强或曝光过度")
        
        debug_info['suggestions'] = suggestions
        debug_info['analyzer_type'] = type(predictor.analyzer).__name__ if predictor.analyzer else 'None'
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        logger.error(f"调试分析失败: {e}")
        return jsonify({'error': f'调试失败: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


def create_app(config=None):
    """应用工厂函数"""
    if config:
        app.config.update(config)
    return app


if __name__ == '__main__':
    print("启动PBR材质识别系统API服务器...")
    print(f"使用设备: {device}")
    print(f"模型状态: {'已加载' if predictor.model else '未加载'}")
    
    # 尝试多个端口 (关闭debug模式以避免冲突)
    ports = [5001, 5002, 5003, 5004]
    for port in ports:
        try:
            print(f"尝试在端口 {port} 启动服务器...")
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False  # 关闭debug模式避免端口冲突
            )
            print(f"✅ API服务器成功在端口 {port} 启动")
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