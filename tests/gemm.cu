#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void kernel_standard_tiled_gemm(
    float *O, float *X, float *Transform, int m, int n, int k, int TILE_SIZE) {
    /*
        - m represents the independent dimension of the input matrix
        - n represents the independent dimenion of the transformation matrix
        - k represents the common dimension of the 2 matrices
        - Within each kernel, the output is computed as: O = matmul(X, Transform)
        - Transposing the transformation tensor is not required as virtual indexing allows
          for intended navigation along rows and columns of either tensors
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
    for (int t = 0; t < ((k + TILE_SIZE - 1) / TILE_SIZE); ++t) {
        // Load tile of X into shared memory
        if (row < m && (t * TILE_SIZE + threadIdx.x) < k) {
            int X_idx = row * k + t * TILE_SIZE + threadIdx.x;
            X_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = X[X_idx];
        } else {
            X_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Load tile of Transform into shared memory
        if (col < n && (t * TILE_SIZE + threadIdx.x) < k) {
            int T_idx = col * k + t * TILE_SIZE + threadIdx.x;
            T_shmem[threadIdx.x * TILE_SIZE + threadIdx.y] = Transform[T_idx];
        } else {
            T_shmem[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
        }
        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += X_shmem[threadIdx.y * TILE_SIZE + i] * T_shmem[i * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < n) {
        O[row * n + col] = value;
    }

    return;
}

int main() {
    int m = 2;
    int n = 3;
    int k = 3;

    float X[m * k] = {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f};

    float Transform[n * k] = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f};

    float Output[m * n] = {0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f};

    float *d_X, *d_Transform, *d_Output;

    cudaMalloc((void **)&d_X, m * k * sizeof(float));
    cudaMalloc((void **)&d_Transform, n * k * sizeof(float));
    cudaMalloc((void **)&d_Output, m * n * sizeof(float));

    cudaMemcpy(d_X, X, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Transform, Transform, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Output, Output, m * n * sizeof(float), cudaMemcpyHostToDevice);

    int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (n + TILE_SIZE - 1) / TILE_SIZE,
        (m + TILE_SIZE - 1) / TILE_SIZE);

    size_t shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    kernel_standard_tiled_gemm<<<grid, block, shared_mem>>>(
        d_Output, d_X, d_Transform,
        m, n, k, TILE_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(Output, d_Output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f, ", Output[i * j]);
        }
        printf("\n");
    }

    return 0;
}