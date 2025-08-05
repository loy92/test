#!/bin/bash

# PBR材质识别系统状态检查脚本
echo "🔍 PBR材质识别系统状态检查"
echo "=================================================="

# 检查API服务器
echo "🔧 API服务器状态:"
if curl -s http://localhost:5001/ > /dev/null; then
    echo "✅ API服务器运行正常 (http://localhost:5001)"
    API_RESPONSE=$(curl -s http://localhost:5001/ | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'版本: {data.get(\"version\", \"未知\")}')")
    echo "📊 $API_RESPONSE"
else
    echo "❌ API服务器未运行"
fi

echo ""

# 检查Web服务器
echo "🌐 Web服务器状态:"
if curl -s http://localhost:8082/ > /dev/null; then
    echo "✅ Web服务器运行正常 (http://localhost:8082)"
    echo "📱 主页面: http://localhost:8082/index.html"
    echo "🧪 测试页面: http://localhost:8082/test_filename_detection.html"
else
    echo "❌ Web服务器未运行"
fi

echo ""

# 检查端口占用
echo "🔌 端口占用情况:"
if lsof -i :5001 > /dev/null 2>&1; then
    PID=$(lsof -ti :5001)
    echo "✅ 端口5001被进程 $PID 占用 (API服务器)"
else
    echo "❌ 端口5001未被占用"
fi

if lsof -i :8082 > /dev/null 2>&1; then
    PID=$(lsof -ti :8082)
    echo "✅ 端口8082被进程 $PID 占用 (Web服务器)"
else
    echo "❌ 端口8082未被占用"
fi

echo ""
echo "==================================================" 