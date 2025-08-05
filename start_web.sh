#!/bin/bash

# PBR材质识别系统Web服务器启动脚本
echo "🌐 启动PBR材质识别系统Web服务器..."

# 检查端口8082是否被占用
if lsof -i :8082 > /dev/null 2>&1; then
    echo "⚠️  端口8082已被占用，正在清理..."
    PID=$(lsof -ti :8082)
    if [ ! -z "$PID" ]; then
        kill -9 $PID
        echo "✅ 已清理占用端口的进程"
        sleep 2
    fi
fi

# 启动Web服务器
echo "🌐 启动Web服务器 (http://localhost:8082)..."
python3 -m http.server 8082 --directory web &

# 等待服务器启动
sleep 2

# 检查服务器是否成功启动
if curl -s http://localhost:8082/ > /dev/null; then
    echo "✅ Web服务器启动成功！"
    echo "🌐 Web地址: http://localhost:8082"
    echo "📱 主页面: http://localhost:8082/index.html"
    echo "🧪 测试页面: http://localhost:8082/test_filename_detection.html"
    echo ""
    echo "💡 提示: 按 Ctrl+C 停止Web服务器"
    
    # 保持脚本运行
    wait
else
    echo "❌ Web服务器启动失败"
    exit 1
fi 