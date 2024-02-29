#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

int main() {
    const size_t dataSize = 1024 * 972 * 8; // 例如，256MB
    char* hostData;
    char* deviceData;

    cudaMallocHost(&hostData, dataSize); // 使用页锁定内存
    cudaMalloc(&deviceData, dataSize);

    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Transfer time: " << duration.count() << " ms\n";

    cudaFreeHost(hostData);
    cudaFree(deviceData);

    return 0;
}