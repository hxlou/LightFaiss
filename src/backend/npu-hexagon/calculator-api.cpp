#include "calculator-api.h"
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef/AEEStdErr.h>
#include <android/log.h>
#include "rpcmem.h"
#include "remote.h"
#include "calculator.h"
#include "os_defines.h"

using namespace std;

const char *TAG = "calculator";

// This function sets DSP_LIBRARY_PATH environment variable
int calculator_init(const char* dsp_library_path) {
    if (setenv("DSP_LIBRARY_PATH", dsp_library_path, 1) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to set DSP_LIBRARY_PATH");
        return -1;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "DSP_LIBRARY_PATH set to: %s", dsp_library_path);
    return 0;
}


int64 calculator_sum_cpp(const int* vec, int len) {
    remote_handle64 handle = 0;
    char* uri;
    int alloc_len = 0;

    Calculator *object = new Calculator();
    object->result = 0;
    object->nErr = 0;
    int *test = 0;
    int64 result = 0;

    alloc_len = sizeof(*test) * len;
    __android_log_print(ANDROID_LOG_INFO, TAG, "Allocating %d bytes", alloc_len);
    if (0 == (test = (int*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, alloc_len))) {
        object->nErr = 1;
        __android_log_print(ANDROID_LOG_ERROR, TAG, "rpcmem_alloc failed with nErr = 0x%x", object->nErr);
        goto bail;
    }
    for (int i = 0; i < len; i++) {
        test[i] = vec[i];
    }

    __android_log_print(ANDROID_LOG_INFO, TAG, "Opening handle");
    uri = (char*)calculator_URI "&_dom=cdsp";
    if(remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.enable = 1;
        data.domain = CDSP_DOMAIN_ID;
        if (0 != (object->nErr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data)))) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "remote_session_control failed for CDSP, returned 0x%x", object->nErr);
            goto bail;
        }
        __android_log_print(ANDROID_LOG_INFO, TAG, "Requested signature-free dynamic module offload");
    } else {
        object->nErr = AEE_EUNSUPPORTED;
        __android_log_print(ANDROID_LOG_ERROR, TAG, "remote_session_control interface is not supported on this device, returned 0x%x", object->nErr);
        goto bail;
    }
    object->nErr = calculator_open(uri, &handle);
    if (object->nErr != 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Handle open failed, returned 0x%x", object->nErr);
        goto bail;
    }

    __android_log_print(ANDROID_LOG_INFO, TAG, "Call calculator_sum");
    if (0 == (object->nErr = calculator_sum(handle, test, len, &object->result))) {
        result = object->result;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "calculator_sum call returned err 0x%x, result %lld", object->nErr, object->result);

bail:
    if (handle) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Closing handle");
        object->nErr = calculator_close(handle);
        if (object->nErr != 0) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Handle close failed, returned 0x%x", object->nErr);
        }
    }
    if (test) {
        rpcmem_free(test);
    }
    if (object) {
        delete object;
    }
    return result;
}

int calculator_gemm_cpp(const float* matrix1,
						const float* matrix2,
						uint32_t m, uint32_t k, uint32_t n,
						float* output_matrix,
						bool transX, bool transY) {

    remote_handle64 handle = 0;
    char* uri = nullptr;
    int nErr = 0;
    int ret = 0;

    float* dsp_matrix1 = nullptr;
    float* dsp_matrix2 = nullptr;
    float* dsp_output = nullptr;

    size_t matrix1_bytes = m * k * sizeof(float);
    size_t matrix2_bytes = k * n * sizeof(float);
    size_t output_bytes = m * n * sizeof(float);

    __android_log_print(ANDROID_LOG_INFO, TAG, "GEMM: m=%u, k=%u, n=%u", m, k, n);

    // 分配 ION 内存
    __android_log_print(ANDROID_LOG_INFO, TAG, "GEMM: Allocating ION memory for inputs and output...");
    dsp_matrix1 = (float*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, matrix1_bytes);
    if (!dsp_matrix1) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: rpcmem_alloc failed for matrix1.");
        ret = -1;
        goto bail;
    }

    dsp_matrix2 = (float*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, matrix2_bytes);
    if (!dsp_matrix2) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: rpcmem_alloc failed for matrix2.");
        ret = -1;
        goto bail;
    }
    
    dsp_output = (float*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, output_bytes);
    if (!dsp_output) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: rpcmem_alloc failed for output.");
        ret = -1;
        goto bail;
    }
    memcpy(dsp_matrix1, matrix1, matrix1_bytes);
    memcpy(dsp_matrix2, matrix2, matrix2_bytes);

    __android_log_print(ANDROID_LOG_INFO, TAG, "GEMM: Opening handle...");
    uri = (char*)calculator_URI "&_dom=cdsp";
    // TODO: remote_session_control 应该只需要在进程生命周期内调用一次，
    if(remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.enable = 1;
        data.domain = CDSP_DOMAIN_ID;
        if (0 != (nErr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data)))) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: remote_session_control failed, returned 0x%x", nErr);
            ret = -1;
            goto bail;
        }
    } else {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: remote_session_control is not supported.");
        ret = -1;
        goto bail;
    }

    nErr = calculator_open(uri, &handle);
    if (nErr != 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: Handle open failed, returned 0x%x", nErr);
        ret = -1;
        goto bail;
    }

    __android_log_print(ANDROID_LOG_INFO, TAG, "GEMM: Calling remote function calculator_gemm...");
    nErr = calculator_gemm(handle,
							dsp_matrix1, m * k,
							dsp_matrix2, k * n,
							dsp_output, m * n,
							m, k, n,
							transX, transY);
                           
    if (nErr != 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Error: calculator_gemm call failed, returned 0x%x", nErr);
        ret = -1;
    } else {
        __android_log_print(ANDROID_LOG_INFO, TAG, "GEMM: Remote call successful.");
        memcpy(output_matrix, dsp_output, output_bytes);
    }


bail:
    __android_log_print(ANDROID_LOG_INFO, TAG, "GEMM: Cleaning up resources...");
    if (handle) {
        if (calculator_close(handle) != 0) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "GEMM Warning: Handle close failed.");
        }
    }
    if (dsp_matrix1) rpcmem_free(dsp_matrix1);
    if (dsp_matrix2) rpcmem_free(dsp_matrix2);
    if (dsp_output) rpcmem_free(dsp_output);

    return ret;
}

int test_calculator() {
    __android_log_print(ANDROID_LOG_INFO, TAG, "=== NPU Calculator 示例 ===");
    
    // 1. 初始化DSP环境
    const char* dsp_path = "/data/local/tmp";
    if (calculator_init(dsp_path) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "初始化失败");
        return -1;
    }
    
    // 2. 准备测试数据
    vector<int> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 构建输入向量的字符串表示用于日志
    char input_str[256] = "输入向量: ";
    char temp[16];
    for (int val : test_data) {
        sprintf(temp, "%d ", val);
        strcat(input_str, temp);
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "%s", input_str);
    
    // 3. 调用NPU计算sum
    int64 result = calculator_sum_cpp(test_data.data(), test_data.size());
    
    __android_log_print(ANDROID_LOG_INFO, TAG, "NPU计算结果: %lld", result);
    
    // 4. 验证结果
    int64 expected = 0;
    for (int val : test_data) {
        expected += val;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "期望结果: %lld", expected);
    __android_log_print(ANDROID_LOG_INFO, TAG, "计算正确: %s", (result == expected ? "是" : "否"));
    
    return 0;
}
