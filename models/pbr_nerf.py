"""
基于Nerfies的PBR材质参数估计模型
扩展原始Nerfies以输出PBR材质参数：金属度、粗糙度、透明度、凹凸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """位置编码模块，用于提高神经网络对高频细节的表达能力"""
    
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # 创建频率
        freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入坐标 [..., D]
        Returns:
            编码后的特征 [..., encoded_dim]
        """
        output = []
        if self.include_input:
            output.append(x)
            
        for freq in self.freq_bands:
            output.append(torch.sin(freq * x))
            output.append(torch.cos(freq * x))
            
        return torch.cat(output, dim=-1)


class MLPLayer(nn.Module):
    """多层感知机层"""
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PBRNeRF(nn.Module):
    """
    基于Nerfies的PBR材质参数估计网络
    输出：密度、颜色、金属度、粗糙度、透明度、法线
    """
    
    def __init__(
        self,
        pos_encoding_freqs: int = 10,
        dir_encoding_freqs: int = 4,
        density_layers: int = 8,
        density_hidden_dim: int = 256,
        pbr_layers: int = 4,
        pbr_hidden_dim: int = 128,
        skip_connections: Tuple[int, ...] = (4,),
        use_viewdirs: bool = True,
        deformation_layers: int = 6,
        deformation_hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.use_viewdirs = use_viewdirs
        self.skip_connections = skip_connections
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(pos_encoding_freqs)
        self.dir_encoder = PositionalEncoding(dir_encoding_freqs)
        
        # 计算编码维度
        pos_encoded_dim = 3 + 2 * 3 * pos_encoding_freqs
        dir_encoded_dim = 3 + 2 * 3 * dir_encoding_freqs if use_viewdirs else 0
        
        # 变形网络（基于Nerfies的核心创新）
        self.deformation_net = nn.ModuleList()
        in_dim = pos_encoded_dim + 3  # 位置编码 + 时间/变形参数
        
        for i in range(deformation_layers):
            out_dim = deformation_hidden_dim if i < deformation_layers - 1 else 3
            self.deformation_net.append(MLPLayer(in_dim, out_dim))
            in_dim = deformation_hidden_dim
            
        # 密度网络
        self.density_net = nn.ModuleList()
        in_dim = pos_encoded_dim
        
        for i in range(density_layers):
            out_dim = density_hidden_dim
            if i == density_layers - 1:
                out_dim = density_hidden_dim + 1  # +1 for density
                
            self.density_net.append(MLPLayer(in_dim, out_dim))
            
            # 跳跃连接
            if i in skip_connections:
                in_dim = density_hidden_dim + pos_encoded_dim
            else:
                in_dim = density_hidden_dim
                
        # PBR材质参数网络
        pbr_input_dim = density_hidden_dim
        if use_viewdirs:
            pbr_input_dim += dir_encoded_dim
            
        self.pbr_net = nn.ModuleList()
        in_dim = pbr_input_dim
        
        for i in range(pbr_layers):
            if i == pbr_layers - 1:
                # 输出：RGB(3) + 金属度(1) + 粗糙度(1) + 透明度(1) + 法线(3) = 9
                out_dim = 9
            else:
                out_dim = pbr_hidden_dim
                
            self.pbr_net.append(MLPLayer(in_dim, out_dim))
            in_dim = pbr_hidden_dim
            
    def forward(
        self, 
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
        deformation_code: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            positions: 3D位置 [..., 3]
            directions: 视线方向 [..., 3] (可选)
            deformation_code: 变形编码 [..., 3] (可选，用于动态场景)
        
        Returns:
            包含density, rgb, metallic, roughness, transparency, normal的字典
        """
        # 位置编码
        pos_encoded = self.pos_encoder(positions)
        
        # 变形网络（Nerfies核心）
        if deformation_code is not None:
            deform_input = torch.cat([pos_encoded, deformation_code], dim=-1)
            deformed_pos = positions
            
            x = deform_input
            for i, layer in enumerate(self.deformation_net):
                x = layer(x)
                if i < len(self.deformation_net) - 1:
                    x = F.relu(x)
                    
            # 应用变形
            deformed_pos = positions + x
            pos_encoded = self.pos_encoder(deformed_pos)
        
        # 密度网络
        x = pos_encoded
        density_features = None
        
        for i, layer in enumerate(self.density_net):
            if i in self.skip_connections:
                x = torch.cat([x, pos_encoded], dim=-1)
                
            x = layer(x)
            
            if i == len(self.density_net) - 1:
                # 最后一层分离密度和特征
                density = x[..., 0]
                density_features = x[..., 1:]
            else:
                x = F.relu(x)
        
        # PBR材质参数网络
        pbr_input = density_features
        
        if self.use_viewdirs and directions is not None:
            dir_encoded = self.dir_encoder(directions)
            pbr_input = torch.cat([pbr_input, dir_encoded], dim=-1)
            
        x = pbr_input
        for i, layer in enumerate(self.pbr_net):
            x = layer(x)
            if i < len(self.pbr_net) - 1:
                x = F.relu(x)
        
        # 解析输出
        rgb = torch.sigmoid(x[..., :3])  # RGB颜色
        metallic = torch.sigmoid(x[..., 3:4])  # 金属度 [0,1]
        roughness = torch.sigmoid(x[..., 4:5])  # 粗糙度 [0,1]
        transparency = torch.sigmoid(x[..., 5:6])  # 透明度 [0,1]
        normal = F.normalize(x[..., 6:9], dim=-1)  # 法线向量
        
        return {
            'density': F.relu(density),
            'rgb': rgb,
            'metallic': metallic,
            'roughness': roughness,
            'transparency': transparency,
            'normal': normal
        }


class PBRLoss(nn.Module):
    """PBR材质参数估计的损失函数"""
    
    def __init__(
        self,
        rgb_weight: float = 1.0,
        metallic_weight: float = 0.5,
        roughness_weight: float = 0.5,
        transparency_weight: float = 0.3,
        normal_weight: float = 0.7,
        density_weight: float = 0.1
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.metallic_weight = metallic_weight
        self.roughness_weight = roughness_weight
        self.transparency_weight = transparency_weight
        self.normal_weight = normal_weight
        self.density_weight = density_weight
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        计算PBR损失
        
        Args:
            predictions: 模型预测结果
            targets: 目标值
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # RGB损失
        if 'rgb' in targets:
            rgb_loss = F.mse_loss(predictions['rgb'], targets['rgb'])
            losses['rgb_loss'] = rgb_loss
            total_loss += self.rgb_weight * rgb_loss
            
        # 金属度损失
        if 'metallic' in targets:
            metallic_loss = F.mse_loss(predictions['metallic'], targets['metallic'])
            losses['metallic_loss'] = metallic_loss
            total_loss += self.metallic_weight * metallic_loss
            
        # 粗糙度损失
        if 'roughness' in targets:
            roughness_loss = F.mse_loss(predictions['roughness'], targets['roughness'])
            losses['roughness_loss'] = roughness_loss
            total_loss += self.roughness_weight * roughness_loss
            
        # 透明度损失
        if 'transparency' in targets:
            transparency_loss = F.mse_loss(predictions['transparency'], targets['transparency'])
            losses['transparency_loss'] = transparency_loss
            total_loss += self.transparency_weight * transparency_loss
            
        # 法线损失
        if 'normal' in targets:
            normal_loss = 1.0 - F.cosine_similarity(
                predictions['normal'], targets['normal'], dim=-1
            ).mean()
            losses['normal_loss'] = normal_loss
            total_loss += self.normal_weight * normal_loss
            
        # 密度正则化损失
        if 'density' in predictions:
            density_reg = torch.mean(predictions['density'] ** 2)
            losses['density_reg'] = density_reg
            total_loss += self.density_weight * density_reg
            
        losses['total_loss'] = total_loss
        return losses


def create_pbr_nerf_model(config: Dict) -> PBRNeRF:
    """根据配置创建PBR-NeRF模型"""
    return PBRNeRF(
        pos_encoding_freqs=config.get('pos_encoding_freqs', 10),
        dir_encoding_freqs=config.get('dir_encoding_freqs', 4),
        density_layers=config.get('density_layers', 8),
        density_hidden_dim=config.get('density_hidden_dim', 256),
        pbr_layers=config.get('pbr_layers', 4),
        pbr_hidden_dim=config.get('pbr_hidden_dim', 128),
        skip_connections=tuple(config.get('skip_connections', [4])),
        use_viewdirs=config.get('use_viewdirs', True),
        deformation_layers=config.get('deformation_layers', 6),
        deformation_hidden_dim=config.get('deformation_hidden_dim', 128)
    ) 