# PBR材质识别系统 - 基于Nerfies

🎯 **基于Google Nerfies神经辐射场的智能PBR材质参数识别系统**

本系统扩展了原始Nerfies模型，专门用于从单张或多张图片中自动识别材质的物理渲染参数（PBR），包括金属度、粗糙度、透明度和凹凸等属性。

## ✨ 主要特性

- 🧠 **基于Nerfies的神经网络**: 利用可变形神经辐射场技术进行精确的材质参数估计
- 📸 **智能图像分析**: 支持JPG、PNG、WebP格式的图片上传和分析  
- ⚙️ **PBR参数输出**: 自动检测金属度、粗糙度、透明度、法线凹凸强度
- 🎨 **实时可视化**: 材质球实时预览和参数可视化
- 🌐 **现代化界面**: 基于Atomm UI的响应式Web界面
- 🚀 **高性能推理**: 支持GPU加速的快速材质识别
- 📊 **多格式导出**: 支持JSON、CSV、XML格式的结果导出

## 🚀 快速开始

### 系统要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少4GB内存
- 现代浏览器 (Chrome, Firefox, Safari)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd pbr_nerfies_system
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动系统**
```bash
python scripts/start_system.py
```

系统将自动：
- 启动API服务器 (http://localhost:5000)
- 启动Web界面 (http://localhost:8080)  
- 打开浏览器访问界面

### 一键启动

也可以创建桌面快捷方式：
```bash
python scripts/start_system.py --create-shortcut
```

## 📖 使用指南

### 基本使用流程

1. **上传图片**
   - 点击上传区域或拖拽图片文件
   - 支持JPG、PNG、WebP格式
   - 建议分辨率512×512或更高

2. **调整设置**（可选）
   - 采样点数：控制分析精度
   - 渲染质量：平衡速度与质量
   - 启用变形处理：适用于动态场景

3. **开始分析**
   - 点击"开始分析材质"按钮
   - 系统将显示实时进度
   - 分析通常需要5-30秒

4. **查看结果**
   - 查看PBR参数数值和可视化
   - 材质球实时预览效果
   - 预测的材质类型

5. **导出结果**
   - 支持JSON、CSV、XML格式
   - 可包含可视化图像
   - 一键下载分析报告

### 高级功能

#### 模型配置
```json
{
  "numSamples": 64,        // 采样点数 (32-128)
  "renderQuality": "balanced",  // fast/balanced/high
  "enableDeformation": true,    // 启用变形处理
  "enableNormalMapping": true,  // 启用法线检测
  "useGPUAcceleration": true   // GPU加速
}
```

#### API调用示例
```javascript
// 调用PBR分析API
const response = await fetch('http://localhost:5000/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: base64ImageData,
    settings: { numSamples: 64 }
  })
});

const result = await response.json();
console.log(result.results.metallic);  // 金属度
```

## 🛠️ 系统架构

### 技术栈
- **后端**: Python + Flask + PyTorch
- **前端**: Vue 3 + Atomm UI  
- **AI模型**: 基于Nerfies的PBR-NeRF
- **图像处理**: OpenCV + Pillow

### 项目结构
```
pbr_nerfies_system/
├── models/              # 神经网络模型
│   ├── pbr_nerf.py     # PBR-NeRF主模型
│   └── ...
├── data/               # 数据处理
│   ├── dataset.py      # 数据集加载器
│   └── ...
├── training/           # 训练模块
│   ├── trainer.py      # 训练器
│   └── ...
├── api/                # API服务器
│   ├── app.py          # Flask应用
│   └── ...
├── web/                # Web前端
│   ├── index.html      # 主页面
│   └── ...
├── scripts/            # 工具脚本
│   ├── start_system.py # 系统启动器
│   ├── train_pbr_model.py # 训练脚本
│   └── ...
├── configs/            # 配置文件
└── requirements.txt    # 依赖列表
```

## 🔬 AI模型详解

### PBR-NeRF架构

本系统基于Nerfies扩展，主要包含：

1. **位置编码器**: 高频细节表达
2. **变形网络**: 处理动态场景（Nerfies核心创新）
3. **密度网络**: 3D场景密度估计
4. **PBR网络**: 材质参数预测
   - 金属度 (Metallic): [0,1]
   - 粗糙度 (Roughness): [0,1] 
   - 透明度 (Transparency): [0,1]
   - 法线强度 (Normal): [0,1]

### 训练数据

支持两种数据格式：
1. **Nerfies格式**: 多视图图像 + 相机参数
2. **合成数据**: 程序生成的测试数据

### 推理过程

1. 图像预处理和光线生成
2. 3D空间采样点计算
3. 神经网络特征提取
4. 体积渲染积分
5. PBR参数聚合输出

## 🎯 模型训练

### 准备数据集

```bash
# 创建演示数据集
python scripts/train_pbr_model.py --create_demo_data

# 使用自定义数据集
python scripts/train_pbr_model.py --data_dir /path/to/your/dataset
```

### 开始训练

```bash
# 使用默认配置训练
python scripts/train_pbr_model.py

# 自定义配置训练
python scripts/train_pbr_model.py \
  --config configs/custom_config.json \
  --epochs 100 \
  --device cuda
```

### 从检查点恢复

```bash
python scripts/train_pbr_model.py \
  --resume experiments/pbr_nerf_experiment/checkpoints/best_model.pth
```

## 📊 API文档

### 主要接口

#### 分析PBR参数
```http
POST /api/analyze
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSk...",
  "settings": {
    "numSamples": 64,
    "enableDeformation": true
  }
}
```

#### 系统信息
```http
GET /api/system/info
```

#### 健康检查
```http
GET /api/health
```

完整API文档请参考：`http://localhost:5000/` (启动后访问)

## 🔧 配置选项

### 模型配置
- `pos_encoding_freqs`: 位置编码频率数
- `density_layers`: 密度网络层数
- `pbr_layers`: PBR网络层数
- `use_viewdirs`: 是否使用视线方向

### 训练配置
- `num_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `optimizer`: 优化器类型

### 渲染配置
- `num_coarse_samples`: 粗采样点数
- `num_fine_samples`: 细采样点数
- `perturb`: 是否添加随机扰动

## ❓ 常见问题

### Q: 系统启动失败怎么办？
A: 请检查：
1. Python版本是否≥3.8
2. 所有依赖是否已安装：`pip install -r requirements.txt`
3. 端口5000和8080是否被占用

### Q: 分析速度很慢？
A: 建议：
1. 使用GPU加速（需要CUDA）
2. 降低采样点数到32-48
3. 选择"快速"渲染质量

### Q: 分析结果不准确？
A: 可以尝试：
1. 提高图片分辨率到512×512以上
2. 使用更清晰、光照均匀的图片
3. 增加采样点数到96-128
4. 训练自定义模型

### Q: 如何添加新材质类型？
A: 需要：
1. 准备该材质的训练数据
2. 修改模型输出层
3. 重新训练模型
4. 更新前端材质分类逻辑

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 开源协议

本项目采用Apache 2.0开源协议 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Google Nerfies](https://github.com/google/nerfies) - 原始Nerfies实现
- [Atomm UI](https://dev-web.makeblock.com/atomm-ui/) - UI组件库
- PyTorch团队 - 深度学习框架

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至：[your-email@example.com]
- 访问项目主页：[your-website.com]

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！ 