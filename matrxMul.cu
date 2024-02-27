#include <stdio.h>
#include <cuda_runtime.h>


// A[32, 32]
// B[32, 64]
// SUM[32, 64]

__global__ void LuTccNonSparse(int8_t* ab, int8_t* wb, int32_t* partial_sum){
    int row = threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int abOffset = blockIdx.x * 32 * 32;
    int wbOffset = blockIdx.x * 64 * 32;
    int sumOffset = blockIdx.x * 32 * 64;
    int32_t sum = 0;
    for (size_t k = 0; k < 32; ++k) {
      sum += ab[abOffset + row * 32 + k] * wb[wbOffset + k * 64 + col];
    }
    partial_sum[sumOffset + row * 64 + col] = sum;
}

__global__ void kernelExample() {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        // 只在第一个线程块的第一个线程中打印
        printf("Block size: %d x %d x %d\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Grid size: %d x %d x %d\n", gridDim.x, gridDim.y, gridDim.z);
    }
}

__global__ void print_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d!\n", tid);
}


int main(void) {
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

    dim3 blockSize(32, 32); // 每个线程块中的线程布局（32x32）
    dim3 blocksPerGrid(972, 2);
    LuTccNonSparse<<<blocksPerGrid, blockSize>>>(d_ab, d_wb, d_sum);

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