# PBR材质识别系统管理指南

## 🚀 快速启动

### 方法1: 一键启动完整系统
```bash
./start_system.sh
```

### 方法2: 分别启动服务
```bash
# 启动API服务器
./start_api.sh

# 启动Web服务器
./start_web.sh
```

## 🛑 停止系统
```bash
./stop_system.sh
```

## 🔍 检查系统状态
```bash
./check_status.sh
```

## 📊 系统功能

### ✅ 已实现功能
- **批量图片上传和分析**
- **文件名材料类型识别** (metal, plastic, wood, glass, fabric, ceramic)
- **PBR参数分析**:
  - 金属度 (0~1浮点数)
  - 粗糙度 (0~1浮点数)
  - 透明度 (0~1浮点数)
  - 置信度 (百分比)
- **CSV结果导出**
- **实时分析进度显示**

### 🌐 访问地址
- **主页面**: http://localhost:8082/index.html
- **测试页面**: http://localhost:8082/test_filename_detection.html
- **API接口**: http://localhost:5001

## 🔧 端口配置
- **API服务器**: 端口5001
- **Web服务器**: 端口8082

## 📝 使用说明

### 1. 上传图片
- 支持批量上传多张图片
- 建议文件名包含材料关键词，如：
  - `metal_steel_surface.jpg`
  - `plastic_rough_texture.png`
  - `wood_oak_grain.webp`

### 2. 查看分析结果
- 系统自动识别文件名中的材料类型
- 显示4个PBR参数：金属度、粗糙度、透明度、置信度
- 数值格式为0~1浮点数（除置信度外）

### 3. 导出结果
- 点击"导出CSV"按钮
- 包含所有分析参数和材料信息

## 🛠️ 故障排除

### 问题1: 端口被占用
```bash
# 检查端口占用
lsof -i :5001
lsof -i :8082

# 强制清理
./stop_system.sh
```

### 问题2: API服务器无响应
```bash
# 重启API服务器
./start_api.sh
```

### 问题3: Web服务器无响应
```bash
# 重启Web服务器
./start_web.sh
```

### 问题4: 分析进度不更新
- 确保API服务器正常运行
- 检查浏览器控制台是否有错误信息
- 尝试刷新页面重新上传

## 📋 脚本说明

| 脚本 | 功能 | 说明 |
|------|------|------|
| `start_system.sh` | 完整系统启动 | 同时启动API和Web服务器 |
| `start_api.sh` | API服务器启动 | 启动PBR分析API服务 |
| `start_web.sh` | Web服务器启动 | 启动Web界面服务 |
| `stop_system.sh` | 系统停止 | 停止所有服务并清理端口 |
| `check_status.sh` | 状态检查 | 检查所有服务运行状态 |

## 🎯 测试建议

1. **基本功能测试**:
   - 上传单张图片测试分析功能
   - 检查文件名材料识别是否正常
   - 验证PBR参数显示格式

2. **批量处理测试**:
   - 上传多张不同材料的图片
   - 检查批量分析进度显示
   - 测试CSV导出功能

3. **稳定性测试**:
   - 长时间运行系统
   - 多次上传和分析
   - 检查内存和CPU使用情况

## 📞 技术支持

如果遇到问题，请按以下步骤排查：

1. 运行 `./check_status.sh` 检查系统状态
2. 查看终端输出的错误信息
3. 尝试重启系统：`./stop_system.sh && ./start_system.sh`
4. 检查浏览器控制台是否有JavaScript错误 