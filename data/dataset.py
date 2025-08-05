"""
PBR材质数据集处理模块
支持从多视图图像中提取PBR参数的数据加载和预处理
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation


class PBRDataset(Dataset):
    """PBR材质数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_scale: int = 1,
        use_deformation: bool = True,
        max_items: Optional[int] = None
    ):
        """
        Args:
            data_dir: 数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            image_scale: 图像缩放因子
            use_deformation: 是否使用变形编码
            max_items: 最大加载数量
        """
        self.data_dir = data_dir
        self.split = split
        self.image_scale = image_scale
        self.use_deformation = use_deformation
        
        # 加载数据集信息
        self._load_dataset_info()
        self._load_cameras()
        self._load_images()
        
        if max_items is not None:
            self.item_ids = self.item_ids[:max_items]
    
    def _load_dataset_info(self):
        """加载数据集信息"""
        dataset_path = os.path.join(self.data_dir, 'dataset.json')
        with open(dataset_path, 'r') as f:
            dataset_info = json.load(f)
        
        if self.split == 'train':
            self.item_ids = dataset_info['train_ids']
        elif self.split == 'val':
            self.item_ids = dataset_info['val_ids']
        else:
            self.item_ids = dataset_info['ids']
            
        # 加载元数据
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # 加载场景信息
        scene_path = os.path.join(self.data_dir, 'scene.json')
        with open(scene_path, 'r') as f:
            self.scene_info = json.load(f)
    
    def _load_cameras(self):
        """加载相机参数"""
        self.cameras = {}
        camera_dir = os.path.join(self.data_dir, 'camera')
        
        for item_id in self.item_ids:
            camera_path = os.path.join(camera_dir, f'{item_id}.json')
            with open(camera_path, 'r') as f:
                camera_data = json.load(f)
            
            # 转换相机参数
            camera = self._parse_camera(camera_data)
            self.cameras[item_id] = camera
    
    def _parse_camera(self, camera_data: Dict) -> Dict:
        """解析相机参数"""
        # 提取相机参数
        orientation = np.array(camera_data['orientation'])
        position = np.array(camera_data['position'])
        focal_length = camera_data['focal_length']
        principal_point = np.array(camera_data['principal_point'])
        image_size = camera_data['image_size']
        
        # 应用场景变换
        scale = self.scene_info['scale']
        center = np.array(self.scene_info['center'])
        
        # 缩放和平移位置
        position = (position - center) * scale
        
        # 构造相机到世界的变换矩阵
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = orientation.T  # 相机的旋转是world-to-camera，需要转置
        camera_to_world[:3, 3] = position
        
        # 内参矩阵
        fx = fy = focal_length / self.image_scale
        cx, cy = principal_point[0] / self.image_scale, principal_point[1] / self.image_scale
        
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return {
            'camera_to_world': camera_to_world,
            'intrinsics': intrinsics,
            'image_size': (image_size[0] // self.image_scale, image_size[1] // self.image_scale)
        }
    
    def _load_images(self):
        """加载图像"""
        self.images = {}
        image_dir = os.path.join(self.data_dir, 'rgb', f'{self.image_scale}x')
        
        for item_id in self.item_ids:
            image_path = os.path.join(image_dir, f'{item_id}.png')
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[item_id] = np.array(image) / 255.0
    
    def get_rays(self, camera: Dict, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成相机光线"""
        intrinsics = camera['intrinsics']
        camera_to_world = camera['camera_to_world']
        
        # 生成像素坐标
        i, j = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing='xy'
        )
        
        # 转换到相机坐标系
        dirs = np.stack([
            (i - intrinsics[0, 2]) / intrinsics[0, 0],
            -(j - intrinsics[1, 2]) / intrinsics[1, 1],
            -np.ones_like(i)
        ], axis=-1)
        
        # 转换到世界坐标系
        rays_d = np.sum(dirs[..., None, :] * camera_to_world[:3, :3], axis=-1)
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        
        # 光线原点
        rays_o = np.broadcast_to(camera_to_world[:3, 3], rays_d.shape)
        
        return rays_o, rays_d
    
    def sample_rays(self, item_id: str, num_rays: int) -> Dict:
        """采样光线"""
        camera = self.cameras[item_id]
        image = self.images.get(item_id)
        height, width = camera['image_size'][1], camera['image_size'][0]
        
        # 生成所有光线
        rays_o, rays_d = self.get_rays(camera, height, width)
        
        # 随机采样像素
        coords = np.stack(np.meshgrid(
            np.arange(width), np.arange(height), indexing='xy'
        ), axis=-1).reshape(-1, 2)
        
        select_indices = np.random.choice(len(coords), size=num_rays, replace=False)
        select_coords = coords[select_indices]
        
        # 提取对应的光线和颜色
        rays_o_select = rays_o[select_coords[:, 1], select_coords[:, 0]]
        rays_d_select = rays_d[select_coords[:, 1], select_coords[:, 0]]
        
        result = {
            'rays_o': rays_o_select,
            'rays_d': rays_d_select,
            'coords': select_coords
        }
        
        # 如果有图像，提取对应的颜色
        if image is not None:
            rgb = image[select_coords[:, 1], select_coords[:, 0]]
            result['rgb'] = rgb
        
        # 添加变形编码
        if self.use_deformation and item_id in self.metadata:
            warp_id = self.metadata[item_id]['warp_id']
            deformation_code = np.zeros(3)  # 简化的变形编码
            deformation_code[0] = warp_id / 100.0  # 归一化
            result['deformation_code'] = np.repeat(
                deformation_code[None, :], num_rays, axis=0
            )
        
        return result
    
    def __len__(self) -> int:
        return len(self.item_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个训练样本"""
        item_id = self.item_ids[idx]
        
        # 采样光线
        sample_data = self.sample_rays(item_id, num_rays=1024)
        
        # 转换为tensor
        result = {}
        for key, value in sample_data.items():
            result[key] = torch.from_numpy(value).float()
        
        result['item_id'] = item_id
        return result


class PBRDataLoader:
    """PBR数据加载器"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        image_scale: int = 4,
        use_deformation: bool = True
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_scale = image_scale
        self.use_deformation = use_deformation
    
    def get_dataloader(self, split: str = 'train', shuffle: bool = True) -> DataLoader:
        """获取数据加载器"""
        dataset = PBRDataset(
            data_dir=self.data_dir,
            split=split,
            image_scale=self.image_scale,
            use_deformation=self.use_deformation
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_synthetic_pbr_data(
    output_dir: str, 
    num_images: int = 100,
    image_size: Tuple[int, int] = (512, 512)
):
    """
    创建合成PBR数据集用于测试
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建目录结构
    os.makedirs(os.path.join(output_dir, 'camera'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rgb', '1x'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rgb', '4x'), exist_ok=True)
    
    # 生成相机参数
    cameras = []
    metadata = {}
    item_ids = []
    
    for i in range(num_images):
        item_id = f'{i:06d}'
        item_ids.append(item_id)
        
        # 随机相机位置（围绕物体）
        theta = 2 * np.pi * i / num_images
        phi = np.pi / 6 + np.random.normal(0, 0.1)
        radius = 3.0 + np.random.normal(0, 0.2)
        
        position = np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
        
        # 相机朝向原点
        up = np.array([0, 0, 1])
        forward = -position / np.linalg.norm(position)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        orientation = np.stack([right, up, forward], axis=0)
        
        # 相机参数
        camera_data = {
            'orientation': orientation.tolist(),
            'position': position.tolist(),
            'focal_length': 800.0,
            'principal_point': [image_size[0] / 2, image_size[1] / 2],
            'skew': 0.0,
            'pixel_aspect_ratio': 1.0,
            'radial_distortion': [0.0, 0.0, 0.0],
            'tangential': [0.0, 0.0],
            'image_size': list(image_size)
        }
        
        # 保存相机文件
        camera_path = os.path.join(output_dir, 'camera', f'{item_id}.json')
        with open(camera_path, 'w') as f:
            json.dump(camera_data, f, indent=2)
        
        # 元数据
        metadata[item_id] = {
            'warp_id': i % 10,  # 10个不同的变形
            'appearance_id': i % 5,  # 5个不同的外观
            'camera_id': 0
        }
        
        # 生成合成图像（简单的彩色噪声）
        image = np.random.rand(image_size[1], image_size[0], 3)
        image = (image * 255).astype(np.uint8)
        
        # 保存图像
        image_1x = Image.fromarray(image)
        image_1x.save(os.path.join(output_dir, 'rgb', '1x', f'{item_id}.png'))
        
        # 4x缩放图像
        image_4x = image_1x.resize((image_size[0] // 4, image_size[1] // 4))
        image_4x.save(os.path.join(output_dir, 'rgb', '4x', f'{item_id}.png'))
    
    # 保存数据集信息
    train_ids = item_ids[:int(0.8 * num_images)]
    val_ids = item_ids[int(0.8 * num_images):]
    
    dataset_info = {
        'count': num_images,
        'num_exemplars': len(train_ids),
        'ids': item_ids,
        'train_ids': train_ids,
        'val_ids': val_ids
    }
    
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 场景信息
    scene_info = {
        'scale': 1.0,
        'center': [0.0, 0.0, 0.0],
        'near': 0.1,
        'far': 10.0
    }
    
    with open(os.path.join(output_dir, 'scene.json'), 'w') as f:
        json.dump(scene_info, f, indent=2)
    
    print(f"合成数据集已创建在: {output_dir}")
    print(f"包含 {num_images} 张图像，训练集: {len(train_ids)}, 验证集: {len(val_ids)}") 