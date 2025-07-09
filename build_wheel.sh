#!/bin/bash
# 构建EdgeVecDB Python wheel包的脚本

set -e

echo "========================================="
echo "Building EdgeVecDB Python Wheel Package"
echo "========================================="

# 检查Python环境
python3 --version
echo "Python executable: $(which python3)"

# 检查并安装构建依赖
echo "Installing build dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install pybind11 numpy cmake ninja

# 清理之前的构建
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# 构建wheel包
echo "Building wheel package..."
python3 setup.py bdist_wheel

# 显示构建结果
echo "========================================="
echo "Build completed successfully!"
echo "Generated wheel packages:"
ls -la dist/
echo "========================================="

echo "To install the package, run:"
echo "  pip install dist/edgevecdb-*.whl"

echo ""
echo "To test the installation, run:"
echo "  python3 -c 'import edgevecdb; print(edgevecdb.__version__)'"
