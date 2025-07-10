#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pybind11 import get_cmake_dir
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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

class CMakeExtension(Extension):
    """
    一个用于CMake构建的setuptools扩展。
    这只是一个标记，用于在build_ext中识别。
    """
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """自定义构建类，使用CMake构建C++扩展"""

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)
        
        # 调试：打印环境变量
        print("=== Environment Variables ===")
        for key in ["ANDROID_NDK", "PREFIX", "HOST", "CFLAGS", "CPPFLAGS", "LDFLAGS"]:
            value = os.environ.get(key, "NOT SET")
            print(f"{key}: {value}")
        print("==============================")
            
        # 设置CMake参数
        # 注意：我们让 CMakeLists.txt 自己管理输出目录结构，
        # 所以不需要在这里设置 CMAKE_LIBRARY_OUTPUT_DIRECTORY。
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DBUILD_PYTHON_BINDINGS=ON",
            f"-DBUILD_TESTS=OFF",
            f"-DUSE_CPU_BLAS=ON",
            f"-DUSE_GPU_KOMP=ON",
            f"-Dpybind11_DIR={get_cmake_dir()}",
        ]
        
        # Android特定设置 (这部分逻辑保持不变)
        if "ANDROID_NDK" in os.environ:
            ndk_path = os.environ["ANDROID_NDK"]
            prefix_dir = os.environ.get("PREFIX")
            
            if not prefix_dir:
                cflags = os.environ.get("CFLAGS", "")
                ldflags = os.environ.get("LDFLAGS", "")
                for flag in cflags.split():
                    if flag.startswith("-I") and "chaquopy" in flag:
                        chaquopy_include = flag[2:]
                        if chaquopy_include.endswith("/include"):
                            prefix_dir = chaquopy_include[:-8]
                            break
                if not prefix_dir:
                    for flag in ldflags.split():
                        if flag.startswith("-L") and "chaquopy" in flag:
                            chaquopy_lib = flag[2:]
                            if chaquopy_lib.endswith("/lib"):
                                prefix_dir = chaquopy_lib[:-4]
                                break
                print(f"Extracted PREFIX from environment: {prefix_dir}")
            
            if prefix_dir:
                python_include_dir = os.path.join(prefix_dir, "include", "python3.12")
                python_library_dir = os.path.join(prefix_dir, "lib")
                python_library = os.path.join(python_library_dir, "libpython3.12.so")
                import numpy
                numpy_include_dir = numpy.get_include()
                
                print(f"Using PREFIX: {prefix_dir}")
                print(f"Python include dir: {python_include_dir}")
                print(f"Python library: {python_library}")
                print(f"NumPy include dir: {numpy_include_dir}")
                
                cmake_args.extend([
                    f"-DANDROID_NDK={ndk_path}",
                    f"-DCMAKE_TOOLCHAIN_FILE={ndk_path}/build/cmake/android.toolchain.cmake",
                    f"-DANDROID_ABI=arm64-v8a",
                    f"-DANDROID_PLATFORM=android-24",
                    f"-DANDROID_STL=c++_shared",
                    f"-DPYTHON_EXECUTABLE={sys.executable}",
                    f"-DPython_INCLUDE_DIRS={python_include_dir}",
                    f"-DPython_LIBRARIES={python_library}",
                    f"-DPython_NumPy_INCLUDE_DIRS={numpy_include_dir}",
                    "-DPython_FOUND=TRUE",
                    "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
                ])
            else:
                print("WARNING: Could not determine Chaquopy prefix, using basic Android settings")
                cmake_args.extend([
                    f"-DANDROID_NDK={ndk_path}",
                    f"-DCMAKE_TOOLCHAIN_FILE={ndk_path}/build/cmake/android.toolchain.cmake",
                    f"-DANDROID_ABI=arm64-v8a",
                    f"-DANDROID_PLATFORM=android-24",
                    f"-DANDROID_STL=c++_shared",
                ])

        build_args = ["--config", "Release", "--parallel"]
        
        # 创建构建目录
        os.makedirs(self.build_temp, exist_ok=True)
        
        # 运行CMake配置
        cmake_cmd = ["cmake", ext.sourcedir] + cmake_args
        print("=== CMake Configure Command ===")
        print(" ".join(cmake_cmd))
        print("=============================")
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)
        
        # --- 修改点 1: 构建所有默认目标 ---
        # 移除 --target 参数，让 CMake 构建所有在 CMakeLists.txt 中定义的目标
        # (包括 edgevecdb_core 和 kp)
        print("=== CMake Build Command ===")
        build_cmd = ["cmake", "--build", "."] + build_args
        print(" ".join(build_cmd))
        print("=========================")
        subprocess.check_call(build_cmd, cwd=self.build_temp)
        
        # --- 修改点 2: 复制整个构建好的包目录 ---
        # CMake 已经将所有文件（.so, __init__.py）放到了正确的包结构中
        # 我们只需要将整个目录复制到 setuptools 的目标位置
        
        # 源目录：在 CMake 构建目录 (self.build_temp) 下的 edgevecdb/ 文件夹
        cmake_output_dir = os.path.join(self.build_temp, "edgevecdb")
        
        # 目标目录：setuptools 的库构建目录 (self.build_lib) 下的 edgevecdb/ 文件夹
        lib_dir = os.path.join(self.build_lib, "edgevecdb")

        print(f"CMake output directory: {cmake_output_dir}")
        print(f"Target library directory: {lib_dir}")

        if not os.path.exists(cmake_output_dir):
            raise RuntimeError(f"CMake output directory not found: {cmake_output_dir}")

        # 确保父目标目录存在
        os.makedirs(self.build_lib, exist_ok=True)

        # 如果目标目录已存在，先删除，确保是干净的复制
        if os.path.exists(lib_dir):
            shutil.rmtree(lib_dir)

        # 复制整个构建好的包结构
        shutil.copytree(cmake_output_dir, lib_dir)
        print(f"Copied complete package from {cmake_output_dir} to {lib_dir}")

        # --- 新增修改：查找并复制 libkompute.so ---
        # 在整个构建目录中搜索 libkompute.so
        # kompute_so_path = None
        # for root, dirs, files in os.walk(self.build_temp):
        #     if "libkompute.so" in files:
        #         kompute_so_path = os.path.join(root, "libkompute.so")
        #         break
        
        # if kompute_so_path:
        #     print(f"Found libkompute.so at: {kompute_so_path}")
        #     # 将其复制到包的根目录，与 edgevecdb_core.so 在一起
        #     shutil.copy2(kompute_so_path, lib_dir)
        #     print(f"Copied libkompute.so to {lib_dir}")
        # else:
        #     # 如果找不到，构建失败，因为运行时会缺少库
        #     raise RuntimeError("Build failed: could not find libkompute.so in build directory.")


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
    
    # find_packages 会找到 src/python/edgevecdb
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    
    # CMakeExtension 作为一个触发器来调用我们的 CMakeBuild 类。
    # 它本身不代表一个具体的文件，而是代表整个CMake项目。
    # 它的名字可以任意，只是一个标记。
    ext_modules=[
        CMakeExtension("edgevecdb._native"), 
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    zip_safe=False,
    # package_data 确保非 .py 文件（如 .so）被包含在 wheel 中
    package_data={
        "edgevecdb": ["*.so", "kp/*.so"],
    },
)