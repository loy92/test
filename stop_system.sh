#!/bin/bash

# PBR材质识别系统停止脚本
echo "🛑 停止PBR材质识别系统..."

# 停止API服务器
echo "🔧 停止API服务器..."
pkill -f "python3 api/enhanced_app.py"
sleep 2

# 停止Web服务器
echo "🌐 停止Web服务器..."
pkill -f "python3 -m http.server 8082"
sleep 2

# 检查端口是否已释放
if lsof -i :5001 > /dev/null 2>&1; then
    echo "⚠️  端口5001仍被占用，强制清理..."
    PID=$(lsof -ti :5001)
    kill -9 $PID
fi

if lsof -i :8082 > /dev/null 2>&1; then
    echo "⚠️  端口8082仍被占用，强制清理..."
    PID=$(lsof -ti :8082)
    kill -9 $PID
fi

echo "✅ 系统已停止"
echo "🌐 所有服务已关闭" 