#!/bin/bash

# PBR材质识别系统完整启动脚本
echo "🚀 PBR材质识别系统 - 完整启动"
echo "=================================================="

# 检查并启动API服务器
echo "🔧 启动API服务器..."
if ./start_api.sh &> /dev/null & then
    echo "✅ API服务器启动成功"
else
    echo "❌ API服务器启动失败"
    exit 1
fi

# 等待API服务器完全启动
sleep 5

# 检查API服务器状态
if curl -s http://localhost:5001/ > /dev/null; then
    echo "✅ API服务器运行正常"
else
    echo "❌ API服务器未响应"
    exit 1
fi

# 检查并启动Web服务器
echo "🌐 启动Web服务器..."
if ./start_web.sh &> /dev/null & then
    echo "✅ Web服务器启动成功"
else
    echo "❌ Web服务器启动失败"
    exit 1
fi

# 等待Web服务器完全启动
sleep 3

# 检查Web服务器状态
if curl -s http://localhost:8082/ > /dev/null; then
    echo "✅ Web服务器运行正常"
else
    echo "❌ Web服务器未响应"
    exit 1
fi

# 打开浏览器
echo "🌐 打开浏览器..."
open http://localhost:8082/index.html
open http://localhost:8082/test_filename_detection.html

echo ""
echo "🎉 系统启动完成！"
echo "=================================================="
echo "🌐 Web界面: http://localhost:8082/index.html"
echo "🧪 测试页面: http://localhost:8082/test_filename_detection.html"
echo "🔧 API接口: http://localhost:5001"
echo ""
echo "📊 系统功能:"
echo "  • 批量图片上传和分析"
echo "  • 文件名材料类型识别"
echo "  • PBR参数分析 (金属度、粗糙度、透明度)"
echo "  • 0~1浮点数格式显示"
echo "  • CSV结果导出"
echo ""
echo "💡 提示: 按 Ctrl+C 停止系统"
echo "=================================================="

# 保持脚本运行
wait 