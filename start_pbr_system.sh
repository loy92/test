#!/bin/bash
# PBR材质识别系统启动脚本

echo "🚀 启动PBR材质识别系统..."

# 检查Python3是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python3"
    exit 1
fi

# 检查必要的Python包
echo "📦 检查Python依赖..."
python3 -c "import flask, flask_cors, PIL, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少必要的Python包，正在安装..."
    pip3 install flask flask-cors pillow numpy
fi

# 启动API服务器
echo "🔧 启动API服务器..."
python3 api/enhanced_app.py &
API_PID=$!

# 等待API服务器启动
echo "⏳ 等待API服务器启动..."
sleep 5

# 测试API服务器
echo "🔍 测试API连接..."
curl -s http://localhost:5001/api/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API服务器启动成功！"
    echo "🌐 API地址: http://localhost:5001"
    
    # 打开网页界面
    echo "🎯 打开网页界面..."
    sleep 2
    open web/enhanced_index.html
    
    echo ""
    echo "🎉 PBR材质识别系统已完全启动！"
    echo ""
    echo "📖 使用说明:"
    echo "   - 网页界面已在浏览器中打开"
    echo "   - 支持批量图片分析"
    echo "   - 支持材料分类识别"
    echo "   - API服务器运行在端口5001"
    echo ""
    echo "🛑 停止系统: 按 Ctrl+C 或运行 'kill $API_PID'"
    echo ""
    
    # 等待用户停止
    echo "系统正在运行中... 按 Ctrl+C 停止"
    wait $API_PID
    
else
    echo "❌ API服务器启动失败"
    kill $API_PID 2>/dev/null
    exit 1
fi