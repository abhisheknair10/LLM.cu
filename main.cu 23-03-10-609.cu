#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void kernel_standard_tiled_gemm(
    __half *O, __half *X, __half *Transform, int m, int n, int k, int TILE_SIZE) {
    /*
        - m represents the independent dimension of the input matrix
        - n represents the independent dimenion of the transformation matrix
        - k represents the common dimension of the 2 matrices
        - Within each kernel, the output is computed as: O = matmul(X, Transform)
        - Transposing the transformation tensor is not required as virtual indexing allows for
            intended navigation along rows and columns of either tensors
        - Order of variables within kernels obey order of computation
    */
    // Kernel start
    //
    extern __shared__ float shared_mem[];
    float *X_shmem = shared_mem;
    float *T_shmem = shared_mem + TILE_SIZE * TILE_SIZE;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Loop over tiles
    float value = 0.0f;
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t += TILE_SIZE) {
        // Load tile of X into shared memory
        if (row < m && t * TILE_SIZE + threadIdx.x < k) {
            int X_idx = row * k + t * TILE_SIZE + threadIdx.x;
            X_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = __half2float(X[X_idx]);
        } else {
            X_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Load tile of Transform into shared memory
        if ((t * TILE_SIZE + threadIdx.y) < k && col < n) {
            int T_idx = col * k + t * TILE_SIZE + threadIdx.y;
            T_shmem[threadIdx.x * TILE_SIZE + threadIdx.y] = __half2float(Transform[T_idx]);
        } else {
            T_shmem[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
        }
        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += X_shmem[threadIdx.y * TILE_SIZE + i] * T_shmem[threadIdx.x * TILE_SIZE + i];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < n) {
        int O_idx = row * n + col;
        O[O_idx] = __float2half(value);
    }

    return;
}

int main() {
    // Initialize host tensors
    int m = 2;
    int n = 3;
    int k = 3;

    __half X_host[6] = {__float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
                        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)};

    __half Transform_host[9] = {__float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
                                __float2half(4.0f), __float2half(5.0f), __float2half(6.0f),
                                __float2half(7.0f), __float2half(8.0f), __float2half(9.0f)};

    // __float2half(7.0f), __float2half(8.0f), __float2half(9.0f)

    __half Result_host[6];

    // Declare device pointers
    __half *d_X, *d_Transform, *d_Result;

    // Allocate memory on the device
    cudaMalloc((void **)&d_X, m * k * sizeof(__half));
    cudaMalloc((void **)&d_Transform, n * k * sizeof(__half));
    cudaMalloc((void **)&d_Result, m * n * sizeof(__half));

    // Copy data from host to device
    cudaMemcpy(d_X, X_host, m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Transform, Transform_host, n * k * sizeof(__half), cudaMemcpyHostToDevice);

    // Launch kernel (adjust block and grid size as needed)
    int TILE_SIZE = 32;

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid(
        (m + TILE_SIZE - 1) / TILE_SIZE,
        (n + TILE_SIZE - 1) / TILE_SIZE);

    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        d_Result, d_X, d_Transform, m, n, k, TILE_SIZE);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(Result_host, d_Result, m * n * sizeof(__half), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_Transform);
    cudaFree(d_Result);

    // Print result
    for (int i = 0; i < (m * n); ++i) {
        printf("%f ", __half2float(Result_host[i]));
    }
    printf("\n");

    return 0;
}