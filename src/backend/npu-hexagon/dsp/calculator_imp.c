/*==============================================================================
  Copyright (c) 2012-2020 Qualcomm Technologies, Inc.
  All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/
#ifndef _DEBUG
#define _DEBUG
#endif

#define THREAD_COUNT 6

#define VTCM_ENABLED 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "HAP_farf.h"
#include "calculator.h"


#include "HAP_perf.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_compute_res.h"

#include "AEEStdErr.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"

// #include "skel.h"

#ifdef __cplusplus
    // restrict not standard in C++
#    if defined(__GNUC__)
#        define GGML_RESTRICT __restrict__
#    elif defined(__clang__)
#        define GGML_RESTRICT __restrict
#    elif defined(_MSC_VER)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT restrict
#    endif
#endif


#define ALIGN_128_BYTE      128
#define VLEN                128

int calculator_open(const char*uri, remote_handle64* handle) {
   void *tptr = NULL;
  /* can be any value or ignored, rpc layer doesn't care
   * also ok
   * *handle = 0;
   * *handle = 0xdeadc0de;
   */
   tptr = (void *)malloc(1);
   *handle = (remote_handle64)tptr;
   assert(*handle);
   return 0;
}

/**
 * @param handle, the value returned by open
 * @retval, 0 for success, should always succeed
 */
int calculator_close(remote_handle64 handle) {
   if (handle)
      free((void*)handle);
   return 0;
}

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define ALIGN_128_BYTE 128
#define FLOATS_PER_VECTOR (128 / sizeof(float))

#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <HAP_farf.h>

#define ALIGN_128 __attribute__((aligned(128)))

#include <hexagon_types.h>
#include <hexagon_protos.h>

#include "qurt.h"
#include "qurt_atomic_ops.h"

static inline int32_t float_to_bits(float input)
{
    union {
        float f;
        int32_t i;
    } fp32 = {.f = input};
    return fp32.i;
}

// 512*512 矩阵乘法, 311ms
// 1024*1024 矩阵乘法, 2733ms
static int32_t
hvx_gemm_vf_fast(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n)
{
    HVX_Vector *iptr1;
    HVX_Vector *optr;
    HVX_Vector sline1, sline2, temp, sout;
    int32_t vectors_in_rounddown = n / 32;
    float *output_row_address;
    float *matrix2_row_address;

    for (int32_t ncnt = 0; ncnt < vectors_in_rounddown; ++ncnt)
    {
        for (uint32_t mcnt = 0; mcnt < m; mcnt++)
        {
            sout = Q6_V_vzero();
            output_row_address = output + (mcnt * n);
            optr = ((HVX_Vector *)output_row_address) + ncnt;

            if (mcnt + 1 < m)
            {
                for (uint32_t i = 0; i < k; i += 32)
                {
                    HVX_Vector *addr = (HVX_Vector *)(input_matrix1 + (mcnt + 1) * k + i);
                    Q6_dcfetch_A(addr);
                }
            }

            for (uint32_t kcnt = 0; kcnt < k; kcnt++)
            {
                matrix2_row_address = input_matrix2 + (kcnt * n);
                iptr1 = ((HVX_Vector *)matrix2_row_address) + ncnt;
                sline1 = *iptr1++;

                sline2 = Q6_V_vsplat_R(float_to_bits(input_matrix1[mcnt * k + kcnt]));

                temp = Q6_Vqf32_vmpy_VsfVsf(sline1, sline2);
                sout = Q6_Vqf32_vadd_Vqf32Vqf32(sout, temp);
            }
            *optr = Q6_Vsf_equals_Vqf32(sout);
        }
    }

    return 0;
}

typedef struct {
    float* A;
    float* B;
    float* C;
    uint32_t M, K, N;
	// int64* res;
	unsigned int *process_A_row; // Matrix A处理的行号
} args_t;

static void thread_entry(void* arg)
{
    args_t* args = (args_t*)arg;
	// *(args->res) = 888;
    hvx_gemm_vf_fast(args->A, args->B, args->C, args->M, args->K, args->N);
}

#if defined(VTCM_ENABLED)
#include "HAP_vtcm_mgr.h"  // 加入 VTCM 接口
#endif

#include "HAP_perf.h"  // 引入计时相关头文件

static inline int handle_vector_in_matrix(float *restrict input_vector1,
											uint32_t input_vector1_len,
											float *restrict input_vector2,
											uint32_t input_vector2_len,
											float *restrict output_vector)
{
	HVX_Vector vector1, sout, temp;
	HVX_Vector vector2 = *(HVX_Vector *)input_vector2;
	HVX_Vector *output_vector_v = (HVX_Vector *)output_vector;
	sout = Q6_V_vzero();
	for (int i = 0; i < input_vector1_len; ++i)
	{
		vector1 = Q6_V_vsplat_R(float_to_bits(input_vector1[i]));
		temp = Q6_Vqf32_vmpy_VsfVsf(vector1, vector2);
        sout = Q6_Vqf32_vadd_Vqf32Vqf32(sout, temp);
	}
	*output_vector_v = Q6_Vsf_equals_Vqf32(sout);
	return 0;
}

// static unsigned int temp_res = 0;

static void process_walker(void *thread_args) {
	args_t *args = (args_t *)thread_args;
	float *A = args->A;
	float *B = args->B;
	float *C = args->C;
	uint32_t M = args->M;
	uint32_t K = args->K;
	uint32_t N = args->N;

	// qurt_atomic_add(&temp_res, 1);
	// return;

	while (1) {
		unsigned int process_A_row = qurt_atomic_add_return(args->process_A_row, 1) - 1;
		if (process_A_row >= M) {
			break;
		}
		for (int i = 0; i < K; i ++) {
			for (int j = 0; j < N / (sizeof(HVX_Vector) / sizeof(float)); j ++) {
				handle_vector_in_matrix(
					A + process_A_row * K,
					K,
					B + i * N + j * (sizeof(HVX_Vector) / sizeof(float)),
					sizeof(HVX_Vector) / sizeof(float),
					C + process_A_row * N + j * (sizeof(HVX_Vector) / sizeof(float))
				);
			}
		}
	}
}

#define STACK_SIZE 1024*4

static inline void multithread_matmul(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n) {

	unsigned int start_time = HAP_perf_get_time_us();

    qurt_thread_t threads[THREAD_COUNT];
    qurt_thread_attr_t attr[THREAD_COUNT];
    args_t args[THREAD_COUNT];
	void* thread_stack_addr[THREAD_COUNT];
	int retcode;

	static unsigned int shared_row_counter = 0;
	shared_row_counter = 0;

    for (int i = 0; i < THREAD_COUNT; i++) {
		thread_stack_addr[i] = malloc(STACK_SIZE);
		assert(thread_stack_addr[i] != NULL);

        qurt_thread_attr_init(&attr[i]);
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "thread%d", i);
		qurt_thread_attr_set_name(&attr[i], thread_name);
		qurt_thread_attr_set_stack_addr(&attr[i], thread_stack_addr[i]);
        qurt_thread_attr_set_stack_size(&attr[i], STACK_SIZE);
        qurt_thread_attr_set_priority(&attr[i], 100);

        args[i].A = input_matrix1;
        args[i].B = input_matrix2;
        args[i].C = output;
        args[i].M = m;
        args[i].K = k;
        args[i].N = n;
		// args[i].res = res;
		args[i].process_A_row = &shared_row_counter;

		retcode = qurt_thread_create(&threads[i], &attr[i], process_walker, &args[i]);
		assert(retcode == QURT_EOK);
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        qurt_thread_join(threads[i], NULL);
		free(thread_stack_addr[i]);
    }

	// for (int i = 0; i < THREAD_COUNT; i++) {
	// 	free(thread_stack_addr[i]);
	// 	free(args[i].process_A_row);
	// }

    unsigned int end_time = HAP_perf_get_time_us();
    unsigned int elapsed_time_ms = (end_time - start_time) / 1000;

    FARF(HIGH, "Multi-threaded hvx_gemm_vf_fast (x6) took %u ms", elapsed_time_ms);
}

// 256*256（无sum中间变量），1669 ms
// 256*256（有sum中间变量），1223 ms
// 512*512（无sum中间变量），12904 ms
// 512*512（有sum中间变量），9331 ms
static inline void matmul(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n) {
	for (int i = 0;i < m; i++) {
		for (int j = 0; j < n; j++) {
			float sum = 0.0f;
			for (int l = 0; l < k; l++) {
				sum += input_matrix1[i * k + l] * input_matrix2[l * n + j];
			}
			output[i * n + j] = sum;
		}
	}
	return;
}


// 512*512 矩阵乘法（无中间变量）, 9162ms
static inline void matmul_ikj(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n) {
	for (int i = 0; i < m; ++i) {
		for (int l = 0; l < k; ++l) {
			float a = input_matrix1[i * k + l]; 
			for (int j = 0; j < n; ++j) {
				output[i * n + j] += a * input_matrix2[l * n + j];
			}
		}
	}
	return;
}

// 256*256 矩阵乘法, 39ms
// 512*512 矩阵乘法, 254ms（VTCM off）
// 512*512 矩阵乘法, 309ms（VTCM on）
// 1024*1024 矩阵乘法, 2817ms（VTCM off）
static inline void matmul_ikj_hvx(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n) {
	for (int i = 0; i < m; ++i) {
		if (i + 1 < m) {
			// 目前来看，没什么优化
			float *next_row = input_matrix1 + (i + 1) * k;
			for (int l = 0; l < k; l += 16) {
				Q6_dcfetch_A(next_row + l);
			}
		}
		for (int l = 0; l < k; ++l) {
			HVX_Vector vector1 = Q6_V_vsplat_R(float_to_bits(input_matrix1[i * k + l]));
			for (int j = 0; j + 31 < n; j += 32) {
				HVX_Vector vector2 = *(HVX_Vector *)(input_matrix2 + l * n + j);
				HVX_Vector *acc = (HVX_Vector *)(output + i * n + j);
				HVX_Vector mul = Q6_Vqf32_vmpy_VsfVsf(vector1, vector2);
				if (l == k - 1) {
					*acc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(*acc, mul));
				} else {
					*acc = Q6_Vqf32_vadd_Vqf32Vqf32(*acc, mul);
				}
			}
		}
	}
	return;
}

static inline void matmul_ikj_hvx_one_block(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n) {
	for (int i = 0; i < m; ++i) {
		if (i + 1 < m) {
			// 目前来看，没什么优化
			float *next_row = input_matrix1 + (i + 1) * k;
			for (int l = 0; l < k; l += 16) {
				Q6_dcfetch_A(next_row + l);
			}
		}

		// void* vtcm_ptr = HAP_request_async_VTCM(k * sizeof(float), 1, 5000);
		// assert(vtcm_ptr != NULL);
		// memcpy(vtcm_ptr, input_matrix1 + i * k, k * sizeof(float));

		for (int l = 0; l < k; ++l) {
			HVX_Vector vector1 = Q6_V_vsplat_R(float_to_bits(input_matrix1[i * k + l]));
			for (int j = 0; j + 31 < n; j += 32) {
				HVX_Vector vector2 = *(HVX_Vector *)(input_matrix2 + l * n + j);
				HVX_Vector *acc = (HVX_Vector *)(output + i * n + j);
				HVX_Vector mul = Q6_Vqf32_vmpy_VsfVsf(vector1, vector2);
				if (l == k - 1) {
					*acc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(*acc, mul));
				} else {
					*acc = Q6_Vqf32_vadd_Vqf32Vqf32(*acc, mul);
				}
			}
		}

		// int ret = HAP_release_VTCM(vtcm_ptr);
		// assert(ret == 0);

	}
	return;
}

// 512*512，全VTCM，346ms
// 512*512，无VTCM，298ms
static inline void matmul_ikj_hvx_blocks(float *restrict input_matrix1,
                 float *restrict input_matrix2,
                 float *restrict output,
                 uint32_t m,
                 uint32_t k,
                 uint32_t n) {
	// void* vtcm_ptr = HAP_request_async_VTCM(k * n * sizeof(float), 1, 5000);
	// assert(vtcm_ptr != NULL);
	// memcpy(vtcm_ptr, input_matrix2, k * n * sizeof(float));
	
	int block_num = 6;
	for (int i = 0; i < block_num; ++i) {
		int start_row = i * (m / block_num);
		int end_row = (i + 1) * (m / block_num);
		if (i == block_num - 1) {
			end_row = m;
		}
		matmul_ikj_hvx_one_block(
			input_matrix1 + start_row * k,
			input_matrix2,
			output + start_row * n,
			end_row - start_row,
			k,
			n
		);
	}

	// int ret = HAP_release_VTCM(vtcm_ptr);
	// assert(ret == 0);
	return;
}

// int calculator_sum(remote_handle64 h, const int* vec, int vecLen, int64* res) {
//     const int ROWS_A = 1;
//     const int COLS_A = 768;
//     const int ROWS_B = 768;
    
//     // 测试参数配置：统一控制colsb的测试范围
//     const int COLSB_START = 100;
//     const int COLSB_END = 10000;
//     const int COLSB_STEP = 500;
    
//     // 测试不同的矩阵尺寸：cols_b 从 COLSB_START 到 COLSB_END，步长为 COLSB_STEP
//     FARF(HIGH, "Matrix multiplication timing test: 1x768 * 768xN");
    
//     char cols_results[1024] = "";
//     char timing_results[1024] = "";
//     int first_result = 1;
//     double final_sum = 0;
    
//     for (int cols_b = COLSB_START; cols_b <= COLSB_END; cols_b += COLSB_STEP) {
//         float* matrixA = NULL;
//         float* matrixB = NULL;
//         float* matrixC = NULL;
//         void* vtcm_block = NULL;
//         int using_vtcm = 0;
        
//         // 计算所需的总内存大小
//         size_t total_bytes = sizeof(float) * (ROWS_A * COLS_A + ROWS_B * cols_b + ROWS_A * cols_b);
        
// #if defined(VTCM_ENABLED)
//         // 尝试分配 VTCM 内存
//         vtcm_block = HAP_request_VTCM((unsigned int)total_bytes, 0);
        
//         if (vtcm_block) {
//             FARF(HIGH, "VTCM allocation succeeded for cols_b=%d: using VTCM for all matrices.", cols_b);
//             matrixA = (float*)vtcm_block;
//             matrixB = matrixA + ROWS_A * COLS_A;
//             matrixC = matrixB + ROWS_B * cols_b;
//             using_vtcm = 1;
//         } else {
//             FARF(HIGH, "VTCM allocation failed for cols_b=%d: falling back to DDR.", cols_b);
//             matrixA = (float*)memalign(128, sizeof(float) * ROWS_A * COLS_A);
//             matrixB = (float*)memalign(128, sizeof(float) * ROWS_B * cols_b);
//             matrixC = (float*)memalign(128, sizeof(float) * ROWS_A * cols_b);
//             assert(matrixA && matrixB && matrixC);
//         }
// #else
//         // VTCM 未启用，使用 DDR
//         matrixA = (float*)memalign(128, sizeof(float) * ROWS_A * COLS_A);
//         matrixB = (float*)memalign(128, sizeof(float) * ROWS_B * cols_b);
//         matrixC = (float*)memalign(128, sizeof(float) * ROWS_A * cols_b);
//         assert(matrixA && matrixB && matrixC);
// #endif
        
//         // 初始化矩阵
//         for (int i = 0; i < ROWS_A * COLS_A; ++i) matrixA[i] = 1.0f;
//         for (int i = 0; i < ROWS_B * cols_b; ++i) matrixB[i] = 1.0f;
//         for (int i = 0; i < ROWS_A * cols_b; ++i) matrixC[i] = 0.0f;
        
//         // 测试矩阵乘法耗时
//         unsigned int start_time = HAP_perf_get_time_us();
//         hvx_gemm_vf_fast(matrixA, matrixB, matrixC, ROWS_A, COLS_A, cols_b);
//         unsigned int end_time = HAP_perf_get_time_us();
//         double elapsed_time_ms = (end_time - start_time) / 1000.0;
        
//         // 计算矩阵乘法结果的元素和
//         double current_sum = 0;
//         for (int i = 0; i < ROWS_A * cols_b; ++i) {
//             current_sum += matrixC[i];
//         }
        
//         // 输出当前计算的元素和
//         FARF(HIGH, "cols_b=%d, timing=%.6fms, sum=%f", cols_b, elapsed_time_ms, current_sum);
        
//         // 将结果添加到汇总字符串中
//         char temp_cols[32];
//         char temp_time[32];
//         snprintf(temp_cols, sizeof(temp_cols), "%s%d", 
//                  first_result ? "" : ",", cols_b);
//         snprintf(temp_time, sizeof(temp_time), "%s%.6f", 
//                  first_result ? "" : ",", elapsed_time_ms);
//         strcat(cols_results, temp_cols);
//         strcat(timing_results, temp_time);
//         first_result = 0;
        
//         // 保存最后一次的结果
//         if (cols_b + COLSB_STEP > COLSB_END) {
//             final_sum = current_sum;
//         }
        
//         // 释放内存
// #if defined(VTCM_ENABLED)
//         if (using_vtcm) {
//             HAP_release_VTCM(vtcm_block);
//         } else {
//             free(matrixA);
//             free(matrixB);
//             free(matrixC);
//         }
// #else
//         free(matrixA);
//         free(matrixB);
//         free(matrixC);
// #endif
//     }
    
//     // 分别输出矩阵列数和对应的时间
//     FARF(HIGH, "%s", cols_results);
//     FARF(HIGH, "%s", timing_results);
    
//     // 返回最后一次矩阵乘法结果的元素和
//     *res = (int64)final_sum;
    
//     return 0;
// }

int calculator_sum(remote_handle64 h, const int* vec, int vecLen, int64* res) {
    #define ROWS_A 2048
    #define COLS_A 2048
    #define ROWS_B 2048
    #define COLS_B 2048

    static float* matrixA = NULL;
    static float* matrixB = NULL;
    static float* matrixC = NULL;
    static int initialized = 0;

    if (!initialized) {
#if defined(VTCM_ENABLED)
        size_t total_bytes = sizeof(float) * (ROWS_A * COLS_A + ROWS_B * COLS_B + ROWS_A * COLS_B);
        void* vtcm_block = HAP_request_VTCM((unsigned int)total_bytes, 0);

		// assert(vtcm_block); // 先只让他存在VTCM

        if (vtcm_block) {
            FARF(HIGH, "VTCM allocation succeeded: using VTCM for all matrices.");
            matrixA = (float*)vtcm_block;
            matrixB = matrixA + ROWS_A * COLS_A;
            matrixC = matrixB + ROWS_B * COLS_B;
        } else {
            FARF(HIGH, "VTCM allocation failed: falling back to DDR.");
            matrixA = (float*)memalign(128, sizeof(float) * ROWS_A * COLS_A);
            matrixB = (float*)memalign(128, sizeof(float) * ROWS_B * COLS_B);
            matrixC = (float*)memalign(128, sizeof(float) * ROWS_A * COLS_B);
			assert(matrixA && matrixB && matrixC);
        }
#else
        matrixA = (float*)memalign(128, sizeof(float) * ROWS_A * COLS_A);
        matrixB = (float*)memalign(128, sizeof(float) * ROWS_B * COLS_B);
        matrixC = (float*)memalign(128, sizeof(float) * ROWS_A * COLS_B);
		assert(matrixA && matrixB && matrixC);
#endif

        for (int i = 0; i < ROWS_A * COLS_A; ++i) matrixA[i] = 1.0f;
        for (int i = 0; i < ROWS_B * COLS_B; ++i) matrixB[i] = 1.0f;
		for (int i = 0; i < ROWS_A * COLS_B; ++i) matrixC[i] = 0.0f;

        initialized = 1;
    }

	unsigned int start_time = HAP_perf_get_time_us();
	matmul_ikj_hvx(matrixA, matrixB, matrixC, ROWS_A, COLS_A, COLS_B);
	unsigned int end_time = HAP_perf_get_time_us();
    unsigned int elapsed_time_ms = (end_time - start_time) / 1000;

	// double sum = 0;
    // for (int i = 0; i < ROWS_A * COLS_B; ++i) sum += matrixC[i];


	*res = elapsed_time_ms;
	// *res = (int64)sum;
    

#if defined(VTCM_ENABLED)
    if ((uintptr_t)matrixA >= 0x10000000 && (uintptr_t)matrixA < 0x20000000) {
        HAP_release_VTCM(matrixA);
        matrixA = matrixB = matrixC = NULL;
        initialized = 0;
    }
#else
	free(matrixA);
	free(matrixB);
	free(matrixC);
	matrixA = matrixB = matrixC = NULL;
	initialized = 0;
#endif

    return 0;
}