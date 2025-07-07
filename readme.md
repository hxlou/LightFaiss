# LightFaiss

该库是一个仿照Faiss实现的更加轻量的向量库，支持插入、查询等功能。本库相较于Faiss最大的改动是，数据由CPU同一管理，各个计算平台会抽象出对应的计算接口来实现查询加速。目前已经实现：

- CPU_BLAS
- GPU_KOMPUTE

可以通过以下步骤编译本项目：
```bash
mkdir build
cd build
cmake ..
make -j8
```

### python支持

在编译后可以在目录`./build/src/python`下发现python模块`lightfaiss`，包含两个子模块`lightfaiss_py`和`kp`，如果希望直接使用kompute接口，通过`import lightfaiss.kp as kp`来使用，否则有关kp的变量在python与Cpp代码之间转换会失败