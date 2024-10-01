#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference.cuh"
#include "llama3/llama3.cuh"

#define CHECK_CUDA_ERROR()                                       \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s in file '%s' in line %i\n",   \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

const int THREADS_PER_BLOCK = 1024;

__constant__ int EMBED_SIZE = 4096;

__device__ int d_NUM_TOKENS;
int h_NUM_TOKENS;

// Kernel to check and print the embeddings
__global__ void check_embedding(__half *fp16_tensor) {
    for (int token_idx = 0; token_idx < d_NUM_TOKENS; token_idx++) {
        printf("Token %d embeddings:\n", token_idx + 1);
        for (int i = 0; i < EMBED_SIZE; i++) {
            float embedding = __half2float(fp16_tensor[token_idx * EMBED_SIZE + i]);
            printf("%f ", embedding);
        }
        printf("\n\n\n\n\n");
    }
}

void printCudaMemoryInfo() {
    size_t free_memory = 0;
    size_t total_memory = 0;

    // Get the amount of free and total memory on the GPU
    cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);

    if (err == cudaSuccess) {
        // Convert memory sizes from bytes to megabytes (MB)
        printf("Free GPU Memory: %.2f MB\n", (float)free_memory / (1024 * 1024));
        printf("Total GPU Memory: %.2f MB\n", (float)total_memory / (1024 * 1024));
    } else {
        printf("Failed to get CUDA memory info: %s\n", cudaGetErrorString(err));
    }
}

void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens) {
    // Set NUM_TOKENS value in device memory
    h_NUM_TOKENS = h_tokens[0] - 1;
    cudaMemcpyToSymbol(d_NUM_TOKENS, &h_NUM_TOKENS, sizeof(int));
    free(h_tokens);

    printCudaMemoryInfo();

    // Order threads into blocks
    int blocks = (h_NUM_TOKENS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tokens_to_embeddings<<<blocks, THREADS_PER_BLOCK>>>(
        llama3_model->embed_tokens->d_fp16_tensor, X->d_fp16_tensor, d_tokens);
    CHECK_CUDA_ERROR();
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();

    // Launch the check_embedding kernel to print the embeddings
    check_embedding<<<1, 1>>>(X->d_fp16_tensor);
    CHECK_CUDA_ERROR();
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();
}

__global__ void tokens_to_embeddings(__half *embed_tokens, __half *fp16_tensor, int *tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // tokens[0] consists of the length of the entire tokens array
    if (idx > 0 && idx < tokens[0]) {
        int managed_offset = idx - 1;

        for (int i = 0; i < EMBED_SIZE; i++) {
            fp16_tensor[(managed_offset * EMBED_SIZE) + i] =
                embed_tokens[(tokens[managed_offset] * EMBED_SIZE) + i];
        }
    }
}
