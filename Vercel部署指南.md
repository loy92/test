# PBR材质识别系统 - Vercel部署指南

## 🚨 问题分析

**原因：** 原始的`requirements.txt`包含大量重型依赖包（PyTorch、JAX、OpenCV等），总大小超过几GB，远超Vercel的部署限制。

## ✅ 解决方案

### 方案一：轻量级Vercel版本（推荐）

我已经为您创建了专门的Vercel部署版本：

#### 📁 文件结构
```
vercel_requirements.txt    # 轻量级依赖
api/vercel_app.py         # 简化的API服务器
vercel.json               # Vercel配置
web/vercel_index.html     # 适配的前端界面
```

#### 🚀 部署步骤

1. **准备部署文件**
   ```bash
   # 将轻量级依赖替换主依赖文件
   cp vercel_requirements.txt requirements.txt
   ```

2. **推送到Git仓库**
   ```bash
   git add .
   git commit -m "Add Vercel lightweight deployment"
   git push origin main
   ```

3. **在Vercel中部署**
   - 登录 [vercel.com](https://vercel.com)
   - 连接您的Git仓库
   - Vercel会自动检测`vercel.json`配置
   - 点击部署

4. **访问部署的应用**
   - API: `https://your-app.vercel.app/api/health`
   - 前端: `https://your-app.vercel.app/web/vercel_index.html`

#### ⚡ 轻量版特性

**包含功能：**
- ✅ 基础PBR参数分析
- ✅ 4种主要材料识别（木材、金属、塑料、玻璃）
- ✅ 文件名智能检测
- ✅ 简化的图像分析算法
- ✅ 快速响应（<5秒）

**不包含功能：**
- ❌ 复杂的NeRF模型
- ❌ 高级图像处理
- ❌ 批量处理
- ❌ 完整的材料数据库

### 方案二：纯前端版本

如果后端依然有问题，我们可以创建纯前端版本：

#### 特点
- 🚀 完全在浏览器中运行
- 📱 基于简化算法的PBR参数估算
- 🎯 支持本地图片分析
- 💰 零服务器成本

### 方案三：其他云平台部署

如果需要完整功能，推荐以下平台：

#### 1. Railway
```bash
# 支持更大的依赖包
# 免费额度：500小时/月
```

#### 2. Render
```bash
# 适合Python应用
# 免费额度：750小时/月
```

#### 3. Heroku
```bash
# 经典PaaS平台
# 需要付费计划
```

#### 4. Google Cloud Run
```bash
# 容器化部署
# 按使用量付费
```

## 🔧 本地测试轻量版

```bash
# 测试轻量级API
python3 api/vercel_app.py

# 打开浏览器测试
open web/vercel_index.html
```

## 📋 部署检查清单

- [ ] 确认`vercel_requirements.txt`只包含必要依赖
- [ ] 测试`api/vercel_app.py`在本地运行正常
- [ ] 验证`vercel.json`配置正确
- [ ] 确保Git仓库包含所有必要文件
- [ ] 在Vercel中正确连接仓库

## 🐛 常见问题

### 1. 依然提示空间不足
```bash
# 进一步精简依赖
echo "flask==2.0.0" > requirements.txt
echo "flask-cors==3.0.0" >> requirements.txt
```

### 2. API路由不工作
```bash
# 检查vercel.json中的路由配置
# 确保API路径正确映射
```

### 3. 图片上传失败
```bash
# 检查文件大小限制
# Vercel函数有50MB限制
```

## 🎯 推荐部署策略

1. **开发/演示阶段：** 使用Vercel轻量版
2. **生产环境：** 使用Railway/Render部署完整版
3. **高并发需求：** 使用Google Cloud Run
4. **成本敏感：** 使用纯前端版本

## 📞 技术支持

如果遇到问题：
1. 检查Vercel部署日志
2. 确认依赖包版本兼容性
3. 验证API端点响应
4. 测试本地版本对比

---
*轻量级版本牺牲了一些功能换取部署便利性，适合快速演示和测试使用。*