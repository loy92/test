#!/bin/bash

# PBR材质识别系统API启动脚本
echo "🚀 启动PBR材质识别系统API..."

# 检查端口5001是否被占用
if lsof -i :5001 > /dev/null 2>&1; then
    echo "⚠️  端口5001已被占用，正在清理..."
    PID=$(lsof -ti :5001)
    if [ ! -z "$PID" ]; then
        kill -9 $PID
        echo "✅ 已清理占用端口的进程"
        sleep 2
    fi
fi

# 检查是否已有API进程在运行
if pgrep -f "python3 api/enhanced_app.py" > /dev/null; then
    echo "⚠️  检测到API进程正在运行，正在停止..."
    pkill -f "python3 api/enhanced_app.py"
    sleep 2
fi

# 启动API服务器
echo "🌐 启动API服务器 (http://localhost:5001)..."
python3 api/enhanced_app.py &

# 等待服务器启动
sleep 3

# 检查服务器是否成功启动
if curl -s http://localhost:5001/ > /dev/null; then
    echo "✅ API服务器启动成功！"
    echo "🌐 API地址: http://localhost:5001"
    echo "📊 测试API: curl http://localhost:5001/"
    echo ""
    echo "💡 提示: 按 Ctrl+C 停止API服务器"
    
    # 保持脚本运行，显示API日志
    echo "📝 API服务器日志:"
    wait
else
    echo "❌ API服务器启动失败"
    exit 1
fi 