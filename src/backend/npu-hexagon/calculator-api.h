#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <stdint.h>
#include <AEEStdDef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file calculator.h
 * @brief NPU Hexagon Calculator 接口定义
 * 
 * 这个头文件定义了使用Qualcomm Hexagon DSP进行计算的C++接口
 */

/**
 * Calculator类，用于封装计算结果和错误状态
 */
class Calculator {
public:
    int64 result;        ///< 计算结果
    int nErr;            ///< 错误代码
    
    Calculator() : result(0), nErr(0) {}
};

/**
 * 初始化DSP环境
 * 
 * @param dsp_library_path DSP库文件路径，通常是 "/data/local/tmp"
 * @return 0 表示成功，-1 表示失败
 * 
 * @note 这个函数必须在调用其他计算函数之前调用
 * @example
 * @code
 * if (calculator_init("/data/local/tmp") != 0) {
 *     printf("DSP初始化失败\n");
 *     return -1;
 * }
 * @endcode
 */
int calculator_init(const char* dsp_library_path);

/**
 * 使用NPU计算向量元素的和
 * 
 * @param vec 输入向量的指针
 * @param len 向量长度
 * @return 向量元素的和，如果计算失败返回0
 * 
 * @note 调用此函数前必须先调用calculator_init()进行初始化
 * @warning 确保vec指针有效且len不超过实际数组大小
 * 
 * @example
 * @code
 * int data[] = {1, 2, 3, 4, 5};
 * int64 result = calculator_sum_cpp(data, 5);
 * printf("Sum result: %lld\n", result);
 * @endcode
 */
int64 calculator_sum_cpp(const int* vec, int len);

/**
 * test_calculator函数，演示NPU Calculator的使用方法
 * 
 * @return 0 表示成功，非0 表示失败
 * 
 * @note 这是一个演示函数，展示了完整的使用流程：
 *       1. 初始化DSP环境
 *       2. 准备测试数据
 *       3. 调用NPU计算
 *       4. 验证结果
 */
int test_calculator();

#ifdef __cplusplus
}
#endif

#endif // CALCULATOR_H
