#include <stdio.h>
#include <cuda_runtime.h>

// // CUDA Kernel function to add elements of two arrays
// __global__ void vector_add(const float *A, const float *B, float *C, int numElements) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < numElements) {
//         C[i] = A[i] + B[i];
//     }
// }

__global__ void TccCalcNonSparse(int8_t* ab, int8_t* wb, int32_t* partial_sum){
    int matrx_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t m = 0; m != 32; ++m) {
      for(size_t n = 0; n != 64; ++n){
         partial_sum[matrx_idx * 2048 + m * 64 + n] = 0;
      }  
      // for (size_t k = 0; k != 32; ++k) {
      //   int8_t tmp = ab[matrx_idx *1024 +m*32 +k];
      //   for (size_t n = 0; n != 64; ++n) {
      //     partial_sum[matrx_idx * 2048 + m * 64 + n] += tmp * wb[matrx_idx *2048 + k* 64 + n];
      //   }
      // }
    }
}

// __global__ void TccCalcNonSparse(int8_t* ab, int8_t* wb, int32_t* partial_sum){
//     int matrx_idx = blockDim.x * blockIdx.x + threadIdx.x;
//     for (size_t m = 0; m != 32; ++m) {
//       for (size_t k = 0; k != 32; ++k) {
//         int8_t tmp = ab[matrx_idx *1024 +m*32 +k];
//         for (size_t n = 0; n != 64; ++n) {
//           partial_sum[m][n] += tmp * wb[k][n];
//         }
//       }
//     }
// }


int main(void) {
    int batchSize = 16;
    int abElements = 1024 * batchSize;
    int wbElements = 2048 * batchSize;
    int sumElements = 2048 * batchSize;
    
    int8_t *h_ab, *h_wb, *d_ab, *d_wb;
    int32_t *h_sum, *d_sum;

    // Allocate host memory
    h_ab = (int8_t *)malloc(abElements);
    h_wb = (int8_t *)malloc(wbElements);
    h_sum = (int32_t *)malloc(sumElements * sizeof(int32_t));


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

    // 复制数据到设备等准备工作...

    // 记录核函数开始执行的时间
    cudaEventRecord(start, 0);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 16;
    int blocksPerGrid = 1;
    TccCalcNonSparse<<<blocksPerGrid, threadsPerBlock>>>(d_ab, d_wb, d_sum);

        // 记录核函数结束执行的时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // 等待事件完成

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("TccCalcNonSparse execution time: %f milliseconds\n", milliseconds);


    // Copy result back to host
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