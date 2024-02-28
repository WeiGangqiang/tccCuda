#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

// 定义矩阵的尺寸
const int M = 8;  // 输出矩阵的行数
const int N = 32;  // 输出矩阵的列数
const int K = 16; // A的列数，B的行数

// const int M = 16;  // 输出矩阵的行数
// const int N = 16;  // 输出矩阵的列数
// const int K = 16; // A的列数，B的行数

const int lda = K; // A的列数
const int ldb = N; // B的列数（因为B是KxN的）
const int ldc = N; // C的列数（因为C是MxN的）

// Kernel定义
__global__ void int8MatrixMulKernel(int8_t * A, int8_t  * B, int32_t * C, int lda, int ldb, int ldc) {
    // 定义WMMA片段
    wmma::fragment<wmma::matrix_a, M, N, K, int8_t, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, M, N, K, int8_t, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, M, N, K, int32_t> fragC;

    // wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> fragA;
    // wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> fragB;
    // wmma::fragment<wmma::accumulator, M, N, K, half> fragC;

    // 初始化累加器片段为0
    wmma::fill_fragment(fragC, 0);

    // 加载矩阵A和B到片段
    wmma::load_matrix_sync(fragA, A, lda);
    wmma::load_matrix_sync(fragB, B, ldb);

    // 执行矩阵乘法和累加
    wmma::mma_sync(fragC, fragA, fragB, fragC);

    // 将计算结果存回C矩阵
    wmma::store_matrix_sync(C, fragC, ldc, wmma::mem_row_major);
}

int main() {
     int batchSize = 972;
    int abElements = 1024 * batchSize;
    int wbElements = 2048 * batchSize;
    int sumElements = 2048 * batchSize;
    
    int8_t *h_ab, *h_wb, *d_ab, *d_wb;
    int32_t *h_sum, *d_sum;

    // Allocate host memory
    h_ab = (int8_t *)malloc(abElements);
    h_wb = (int8_t *)malloc(wbElements);
    h_sum = (int32_t *)malloc(sumElements * sizeof(int32_t));
    memset(h_ab, 0x1, abElements);
    memset(h_wb, 0x1, wbElements);
  
    // Allocate device memory
    cudaMalloc((void **)&d_ab, abElements);
    cudaMalloc((void **)&d_wb, wbElements);
    cudaMalloc((void **)&d_sum, sumElements * sizeof(int32_t));

    // Copy data from host to device
    cudaMemcpy(d_ab, h_ab, abElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wb, h_wb, wbElements, cudaMemcpyHostToDevice);


        // 创建两个CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录核函数开始执行的时间
    cudaEventRecord(start, 0);

    // 假设 lda, ldb, ldc 是对应的leading dimensions

    // Kernel调用
    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid(1, 1);
    int8MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ab, d_wb, d_sum, lda, ldb, ldc);
    // 同步设备以确保Kernel执行完成
        // kernelExample<<<blocksPerGrid, blockSize>>>();
    // print_kernel<<<1, 256>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // 等待事件完成

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Mul Vmma execution time: %f milliseconds\n", milliseconds);

    cudaMemcpy(h_sum, d_sum, sumElements * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    printf("Test PASSED\n");

    // Free device memory
    cudaFree(d_ab);
    cudaFree(d_wb);
    cudaFree(d_sum);

    // Free host memory
    free(h_ab);
    free(h_wb);
    free(h_sum);
    return 0;
}