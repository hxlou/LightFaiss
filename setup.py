#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, find_packages, setup

# 读取版本信息
def get_version():
    version_file = Path(__file__).parent / "VERSION.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0"

# 读取README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""

class CMakeBuild(build_ext):
    """自定义构建类，使用CMake构建C++扩展"""
    
    def build_extension(self, ext):
        # 必须是CMakeExtension实例
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)
        
        # 调试：打印环境变量
        print("=== Environment Variables ===")
        for key in ["ANDROID_NDK", "PREFIX", "HOST", "CFLAGS", "CPPFLAGS", "LDFLAGS"]:
            value = os.environ.get(key, "NOT SET")
            print(f"{key}: {value}")
        print("==============================")
            
        # 设置CMake参数
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(self.build_temp)}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_PYTHON_BINDINGS=ON",
            f"-DBUILD_TESTS=OFF",
            f"-DUSE_CPU_BLAS=ON",
            f"-DUSE_GPU_KOMP=ON",
            f"-Dpybind11_DIR={get_cmake_dir()}",
        ]
        
        # Android特定设置
        if "ANDROID_NDK" in os.environ:
            ndk_path = os.environ["ANDROID_NDK"]
            # 查找 Chaquopy Python 头文件和库
            prefix_dir = os.environ.get("PREFIX")
            
            # 如果 PREFIX 没有设置，尝试从 CFLAGS 或 LDFLAGS 中提取路径
            if not prefix_dir:
                cflags = os.environ.get("CFLAGS", "")
                ldflags = os.environ.get("LDFLAGS", "")
                
                # 从 CFLAGS 中提取 chaquopy 路径
                # 例如: -I/path/to/chaquopy/include
                for flag in cflags.split():
                    if flag.startswith("-I") and "chaquopy" in flag:
                        chaquopy_include = flag[2:]  # 移除 -I
                        if chaquopy_include.endswith("/include"):
                            prefix_dir = chaquopy_include[:-8]  # 移除 /include
                            break
                
                # 如果还没找到，从 LDFLAGS 中提取
                if not prefix_dir:
                    for flag in ldflags.split():
                        if flag.startswith("-L") and "chaquopy" in flag:
                            chaquopy_lib = flag[2:]  # 移除 -L
                            if chaquopy_lib.endswith("/lib"):
                                prefix_dir = chaquopy_lib[:-4]  # 移除 /lib
                                break
                
                print(f"Extracted PREFIX from environment: {prefix_dir}")
            
            if prefix_dir:
                python_include_dir = os.path.join(prefix_dir, "include", "python3.12")
                python_library_dir = os.path.join(prefix_dir, "lib")
                python_library = os.path.join(python_library_dir, "libpython3.12.so")
                
                # 查找 numpy 头文件
                import numpy
                numpy_include_dir = numpy.get_include()
                
                print(f"Using PREFIX: {prefix_dir}")
                print(f"Python include dir: {python_include_dir}")
                print(f"Python library: {python_library}")
                print(f"NumPy include dir: {numpy_include_dir}")
                
                # 验证路径是否存在
                print(f"Python include dir exists: {os.path.exists(python_include_dir)}")
                print(f"Python library exists: {os.path.exists(python_library)}")
                print(f"NumPy include dir exists: {os.path.exists(numpy_include_dir)}")
                
                cmake_args.extend([
                    f"-DANDROID_NDK={ndk_path}",
                    f"-DCMAKE_TOOLCHAIN_FILE={ndk_path}/build/cmake/android.toolchain.cmake",
                    f"-DANDROID_ABI=arm64-v8a",
                    f"-DANDROID_PLATFORM=android-21",
                    f"-DANDROID_STL=c++_shared",
                    f"-DPython_EXECUTABLE={sys.executable}",
                    f"-DPYTHON_EXECUTABLE={sys.executable}",  # 也设置旧的变量名
                    f"-DPython_INCLUDE_DIRS={python_include_dir}",
                    f"-DPYTHON_INCLUDE_DIR={python_include_dir}",  # 也设置旧的变量名
                    f"-DPython_LIBRARIES={python_library}",
                    f"-DPYTHON_LIBRARY={python_library}",  # 也设置旧的变量名
                    f"-DPython_NumPy_INCLUDE_DIRS={numpy_include_dir}",
                    f"-DPYTHON_NUMPY_INCLUDE_DIR={numpy_include_dir}",  # 也设置旧的变量名
                    "-DPython_FOUND=TRUE",
                    "-DPython_Interpreter_FOUND=TRUE",
                    "-DPython_Development_FOUND=TRUE",
                    "-DPython_Development.Module_FOUND=TRUE",
                    "-DPython_NumPy_FOUND=TRUE",
                    "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
                ])
            else:
                print("WARNING: Could not determine Chaquopy prefix, using basic Android settings")
                cmake_args.extend([
                    f"-DANDROID_NDK={ndk_path}",
                    f"-DCMAKE_TOOLCHAIN_FILE={ndk_path}/build/cmake/android.toolchain.cmake",
                    f"-DANDROID_ABI=arm64-v8a",
                    f"-DANDROID_PLATFORM=android-21",
                    f"-DANDROID_STL=c++_shared",
                ])
        
        build_args = ["--config", "Release", "--parallel"]
        
        # 创建构建目录
        os.makedirs(self.build_temp, exist_ok=True)
        
        # 打印将要执行的 CMake 命令
        cmake_cmd = ["cmake", ext.sourcedir] + cmake_args
        print("=== CMake Command ===")
        for i, arg in enumerate(cmake_cmd):
            print(f"{i}: {arg}")
        print("====================")
        
        # 运行CMake配置
        subprocess.check_call(
            cmake_cmd, 
            cwd=self.build_temp
        )
        
        # 运行CMake构建
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "edgevecdb_core"] + build_args,
            cwd=self.build_temp
        )
        
        # 查找并复制生成的库文件
        import shutil
        import glob
        
        # 查找构建的库文件
        patterns = [
            "edgevecdb_core*.so", 
            "edgevecdb_core*.pyd", 
            "edgevecdb_core*.dylib"
        ]
        
        built_lib = None
        for pattern in patterns:
            files = glob.glob(os.path.join(self.build_temp, "**", pattern), recursive=True)
            if files:
                built_lib = files[0]
                break
        
        if built_lib:
            # 确保目标目录存在
            lib_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
            os.makedirs(lib_dir, exist_ok=True)
            
            # 复制文件
            target_path = self.get_ext_fullpath(ext.name)
            shutil.copy2(built_lib, target_path)
            print(f"Copied {built_lib} to {target_path}")
        else:
            raise RuntimeError(f"Could not find built extension library in {self.build_temp}")

class CMakeExtension(Extension):
    """CMake扩展类"""
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

# 主要设置
setup(
    name="edgevecdb",
    version=get_version(),
    author="EdgeVecDB Team",
    author_email="team@edgevecdb.com",
    description="A lightweight vector search library for edge devices",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/edgevecdb",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    ext_modules=[
        CMakeExtension("edgevecdb.edgevecdb_core"),
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    zip_safe=False,
    include_package_data=True,
)
