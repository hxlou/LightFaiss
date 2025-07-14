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

// 演示如何使用NPU计算sum的主函数
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
