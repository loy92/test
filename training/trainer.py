"""
PBR-NeRF训练器
实现完整的训练循环，包括验证、模型保存、日志记录等功能
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, List
import json
from tqdm import tqdm

# 修复导入路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.pbr_nerf import PBRNeRF, PBRLoss, create_pbr_nerf_model
    from data.dataset import PBRDataLoader
except ImportError as e:
    print(f"警告: 无法导入模块 {e}")
    # 创建占位符类
    class PBRNeRF: pass
    class PBRLoss: pass
    class PBRDataLoader: pass
    def create_pbr_nerf_model(config): return None


class PBRTrainer:
    """PBR-NeRF训练器"""
    
    def __init__(
        self,
        config: Dict,
        model: Optional[PBRNeRF] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            config: 训练配置
            model: 预训练模型（可选）
            device: 计算设备
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        if model is None:
            self.model = create_pbr_nerf_model(config['model'])
        else:
            self.model = model
        self.model.to(self.device)
        
        # 创建损失函数
        self.criterion = PBRLoss(**config.get('loss', {}))
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建数据加载器
        self.data_loader = PBRDataLoader(**config['data'])
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'renders'), exist_ok=True)
        
        # 创建日志记录器
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optim_config = self.config['training']['optimizer']
        optim_type = optim_config.get('type', 'adam')
        lr = optim_config.get('lr', 1e-4)
        weight_decay = optim_config.get('weight_decay', 0.0)
        
        if optim_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optim_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optim_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.config['training'].get('scheduler')
        if scheduler_config is None:
            return None
        
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 1000),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 1000)
            )
        else:
            return None
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 移动数据到设备
        rays_o = batch['rays_o'].to(self.device)  # [B, N, 3]
        rays_d = batch['rays_d'].to(self.device)  # [B, N, 3]
        rgb_target = batch['rgb'].to(self.device)  # [B, N, 3]
        
        deformation_code = None
        if 'deformation_code' in batch:
            deformation_code = batch['deformation_code'].to(self.device)
        
        # 前向传播
        B, N = rays_o.shape[:2]
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        if deformation_code is not None:
            deformation_code_flat = deformation_code.reshape(-1, 3)
        else:
            deformation_code_flat = None
        
        # 体积渲染
        predictions = self.volume_render(
            rays_o_flat, rays_d_flat, deformation_code_flat
        )
        
        # 重塑输出
        for key in predictions:
            if predictions[key].dim() == 2:
                predictions[key] = predictions[key].reshape(B, N, -1)
            else:
                predictions[key] = predictions[key].reshape(B, N)
        
        # 计算损失
        targets = {'rgb': rgb_target}
        
        # 如果有PBR标签，添加到目标中
        if 'metallic' in batch:
            targets['metallic'] = batch['metallic'].to(self.device)
        if 'roughness' in batch:
            targets['roughness'] = batch['roughness'].to(self.device)
        if 'transparency' in batch:
            targets['transparency'] = batch['transparency'].to(self.device)
        if 'normal' in batch:
            targets['normal'] = batch['normal'].to(self.device)
        
        losses = self.criterion(predictions, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # 梯度裁剪
        max_grad_norm = self.config['training'].get('max_grad_norm')
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        # 返回损失值
        loss_dict = {}
        for key, value in losses.items():
            loss_dict[key] = value.item()
        
        return loss_dict
    
    def volume_render(
        self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor,
        deformation_code: Optional[torch.Tensor] = None,
        num_samples: int = 64,
        num_fine_samples: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        体积渲染
        
        Args:
            rays_o: 光线原点 [N, 3]
            rays_d: 光线方向 [N, 3]
            deformation_code: 变形编码 [N, 3]
            num_samples: 粗采样点数
            num_fine_samples: 细采样点数
            
        Returns:
            渲染结果字典
        """
        N = rays_o.shape[0]
        near = self.config['data'].get('near', 0.1)
        far = self.config['data'].get('far', 10.0)
        
        # 粗采样
        t_vals = torch.linspace(near, far, num_samples, device=self.device)
        t_vals = t_vals.expand(N, num_samples)
        
        # 添加噪声
        if self.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
        
        # 计算采样点
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        
        # 扩展变形编码
        if deformation_code is not None:
            deformation_code_expanded = deformation_code[:, None, :].expand(-1, num_samples, -1)
            deformation_code_flat = deformation_code_expanded.reshape(-1, 3)
        else:
            deformation_code_flat = None
        
        # 模型预测
        with torch.cuda.amp.autocast(enabled=self.config['training'].get('use_amp', False)):
            outputs = self.model(
                pts_flat,
                rays_d[:, None, :].expand(-1, num_samples, -1).reshape(-1, 3),
                deformation_code_flat
            )
        
        # 重塑输出
        for key in outputs:
            if outputs[key].dim() == 2:
                outputs[key] = outputs[key].reshape(N, num_samples, -1)
            else:
                outputs[key] = outputs[key].reshape(N, num_samples)
        
        # 体积渲染积分
        rendered = self._volume_integration(outputs, t_vals)
        
        return rendered
    
    def _volume_integration(
        self, 
        outputs: Dict[str, torch.Tensor], 
        t_vals: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """体积积分"""
        density = outputs['density']  # [N, S]
        rgb = outputs['rgb']  # [N, S, 3]
        
        # 计算距离
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # 计算透明度
        alpha = 1.0 - torch.exp(-density * dists)
        
        # 计算权重
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        weights = alpha * transmittance
        
        # 渲染RGB
        rgb_rendered = torch.sum(weights[..., None] * rgb, dim=-2)
        
        # 渲染深度
        depth = torch.sum(weights * t_vals, dim=-1)
        
        # 渲染其他PBR参数
        result = {
            'rgb': rgb_rendered,
            'depth': depth,
            'weights': weights
        }
        
        if 'metallic' in outputs:
            result['metallic'] = torch.sum(weights[..., None] * outputs['metallic'], dim=-2)
        if 'roughness' in outputs:
            result['roughness'] = torch.sum(weights[..., None] * outputs['roughness'], dim=-2)
        if 'transparency' in outputs:
            result['transparency'] = torch.sum(weights[..., None] * outputs['transparency'], dim=-2)
        if 'normal' in outputs:
            result['normal'] = torch.sum(weights[..., None] * outputs['normal'], dim=-2)
        
        return result
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        val_loader = self.data_loader.get_dataloader('val', shuffle=False)
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                # 移动数据到设备
                rays_o = batch['rays_o'].to(self.device)
                rays_d = batch['rays_d'].to(self.device)
                rgb_target = batch['rgb'].to(self.device)
                
                deformation_code = None
                if 'deformation_code' in batch:
                    deformation_code = batch['deformation_code'].to(self.device)
                
                # 前向传播
                B, N = rays_o.shape[:2]
                rays_o_flat = rays_o.reshape(-1, 3)
                rays_d_flat = rays_d.reshape(-1, 3)
                
                if deformation_code is not None:
                    deformation_code_flat = deformation_code.reshape(-1, 3)
                else:
                    deformation_code_flat = None
                
                predictions = self.volume_render(
                    rays_o_flat, rays_d_flat, deformation_code_flat
                )
                
                # 重塑输出
                for key in predictions:
                    if predictions[key].dim() == 2:
                        predictions[key] = predictions[key].reshape(B, N, -1)
                    else:
                        predictions[key] = predictions[key].reshape(B, N)
                
                # 计算损失
                targets = {'rgb': rgb_target}
                losses = self.criterion(predictions, targets)
                
                # 累计损失
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
                
                num_batches += 1
        
        # 计算平均损失
        avg_losses = {}
        for key, value in total_losses.items():
            avg_losses[key] = value / num_batches
        
        return avg_losses
    
    def train(self, num_epochs: int):
        """训练模型"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"使用设备: {self.device}")
        
        train_loader = self.data_loader.get_dataloader('train', shuffle=True)
        scheduler = self._create_scheduler()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_losses = {}
            num_batches = 0
            
            # 训练循环
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # 训练步骤
                losses = self.train_step(batch)
                
                # 累计损失
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                
                num_batches += 1
                self.global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'rgb': f"{losses.get('rgb_loss', 0):.4f}"
                })
                
                # 记录训练日志
                if self.global_step % self.config['training'].get('log_interval', 100) == 0:
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                
                # 学习率调度
                if scheduler is not None:
                    scheduler.step()
            
            # 计算平均损失
            avg_losses = {}
            for key, value in epoch_losses.items():
                avg_losses[key] = value / num_batches
            
            print(f"\nEpoch {epoch+1} 训练完成:")
            for key, value in avg_losses.items():
                print(f"  {key}: {value:.6f}")
            
            # 验证
            if epoch % self.config['training'].get('val_interval', 10) == 0:
                print("开始验证...")
                val_losses = self.validate()
                
                print("验证结果:")
                for key, value in val_losses.items():
                    print(f"  val_{key}: {value:.6f}")
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                
                # 保存最佳模型
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint(is_best=True)
                    print("保存最佳模型")
            
            # 定期保存检查点
            if epoch % self.config['training'].get('save_interval', 50) == 0:
                self.save_checkpoint(epoch=epoch)
        
        print("训练完成!")
        self.writer.close()
    
    def save_checkpoint(self, epoch: Optional[int] = None, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            filename = 'best_model.pth'
        elif epoch is not None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        else:
            filename = 'latest_checkpoint.pth'
        
        filepath = os.path.join(self.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"加载检查点成功: epoch {self.current_epoch}, step {self.global_step}")


def create_trainer_from_config(config_path: str, device: str = 'cuda') -> PBRTrainer:
    """从配置文件创建训练器"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return PBRTrainer(config, device=device) 