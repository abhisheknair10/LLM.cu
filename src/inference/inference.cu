#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference.cuh"
#include "llama3/llama3.cuh"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

const int THREADS_PER_BLOCK = 1024;

__constant__ int EMBED_SIZE = 4096;
__device__ int NUM_TOKENS;

// Kernel to check and print the embeddings
__global__ void check_embedding(__half *fp16_tensor, int num_tokens) {
    for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
        printf("Token %d embeddings:\n", token_idx + 1);
        for (int i = 0; i < EMBED_SIZE; i++) {
            float embedding = __half2float(fp16_tensor[token_idx * EMBED_SIZE + i]);
            printf("%f ", embedding);
        }
        printf("\n\n\n\n\n");
    }
}

void inference(Llama3 *llama3_model, Tensor *X, int *tokens, int *h_tokens) {
    // Set NUM_TOKENS value in device memory
    int h_num_tokens = h_tokens[0] - 1;
    cudaMemcpyToSymbol(NUM_TOKENS, &h_num_tokens, sizeof(int));
    free(h_tokens);

    int blocks = (tokens[0] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the token to embedding conversion kernel
    tokens_to_embeddings<<<blocks, THREADS_PER_BLOCK>>>(
        llama3_model->embed_tokens->d_fp16_tensor, X->d_fp16_tensor, tokens);
    cudaDeviceSynchronize();

    // Launch the check_embedding kernel to print the embeddings
    check_embedding<<<1, 1>>>(X->d_fp16_tensor, h_num_tokens);
    cudaDeviceSynchronize();
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
