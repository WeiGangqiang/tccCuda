#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

// #define WMMA_M 16
// #define WMMA_N 16
// #define WMMA_K 16


#define WMMA_M 8
#define WMMA_N 32
#define WMMA_K 16

// // 定义矩阵的尺寸
// const int WMMA_M = 8;  // 输出矩阵的行数
// const int WMMA_M = 32;  // 输出矩阵的列数
// const int K = 16; // A的列数，B的行数


#define WARP_SIZE 32

using namespace nvcuda;

__global__ void wmmaNaiveKernel(const int8_t *  A, const int8_t * B, int32_t * C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = K / WMMA_K;

    const size_t warp_row = blockIdx.x * WMMA_M;
    const size_t warp_col = blockIdx.y * WMMA_N;
    const size_t loopCount = blockIdx.z;
    const size_t loopOffsetA = blockIdx.z * M * K;
    const size_t loopOffsetB = blockIdx.z * N * K;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> C_frag;

    wmma::fill_fragment(C_frag, 0);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + loopOffsetA + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + loopOffsetB + i * WMMA_K + warp_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + loopCount * M * N + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(int8_t *A, int8_t *B, int32_t *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(M/WMMA_M, N/WMMA_N, 972);

    wmmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main(void) {
    int batchSize = 972;
    int abElements = 32 * 32 * batchSize;
    int wbElements = 32 * 64 * batchSize;
    int sumElements = 32 * 64 * batchSize;
    
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

    // dim3 blockSize(32, 32); // 每个线程块中的线程布局（32x32）
    // dim3 blocksPerGrid(972, 2);
    // LuTccNonSparse<<<blocksPerGrid, blockSize>>>(d_ab, d_wb, d_sum);

    wmmaNaive(d_ab, d_wb, d_sum, 32, 64, 32);

    // kernelExample<<<blocksPerGrid, blockSize>>>();
    // print_kernel<<<1, 256>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // 等待事件完成

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("TccCalcNonSparse execution time: %f milliseconds\n", milliseconds);

    cudaMemcpy(h_sum, d_sum, sumElements * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (size_t k = 0; k < sumElements; ++k) {
      if(h_sum[k] != 32){
        printf("Test Failed, h_sum[%ld]=%d\n", k, h_sum[k]);  
        break;  
      }
    }

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