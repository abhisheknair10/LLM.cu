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

    return;
}

/* ******************************** Inference Code ******************************** */

void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens) {
    // Set NUM_TOKENS value in device memory
    h_NUM_TOKENS = h_tokens[0] - 1;
    cudaMemcpyToSymbol(d_NUM_TOKENS, &h_NUM_TOKENS, sizeof(int));
    free(h_tokens);

    tokens_to_embeddings(X, llama3_model, d_tokens);

    // Ahead Of Time memory allocations
    // Allocate once, use everywhere
    Tensor *Q, *K, *V;
    _create_intermediary_attention_tensor(&Q, llama3_model->layers[0]->self_attn_q_proj);
    _create_intermediary_attention_tensor(&K, llama3_model->layers[0]->self_attn_k_proj);
    _create_intermediary_attention_tensor(&V, llama3_model->layers[0]->self_attn_v_proj);

    // Run Inference
    for (int i = 0; i < llama3_model->n_layers; i++) {
        compute_qkv_tensors(Q, K, V, llama3_model->layers[i], X);
    }

    printCudaMemoryInfo();
}

void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens) {
    // Order threads into blocks
    int blocks = (h_NUM_TOKENS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kernel_tokens_to_embeddings<<<blocks, THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, llama3_model->embed_tokens->d_fp16_tensor, d_tokens);

    cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(X->d_fp16_tensor);
    // cudaDeviceSynchronize();
}

__global__ void kernel_tokens_to_embeddings(__half *fp16_tensor, __half *Embed, int *tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // tokens[0] consists of the length of the entire tokens array
    if (idx > 0 && idx <= d_NUM_TOKENS) {
        int managed_offset = idx - 1;
        for (int i = 0; i < EMBED_SIZE; i++) {
            fp16_tensor[(managed_offset * EMBED_SIZE) + i] =
                Embed[(tokens[idx] * EMBED_SIZE) + i];
        }
    }

    return;
}

void _create_intermediary_attention_tensor(Tensor *Attention_Tensor, Tensor *Linear) {
    // Allocate the Tensor struct
    Attention_Tensor = (Tensor *)malloc(sizeof(Tensor));

    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Attention_Tensor->ndim = (int *)malloc(sizeof(int));
    *(Attention_Tensor->ndim) = *(Linear->ndim);

    Attention_Tensor->mem_len = (int *)malloc(sizeof(int));
    *(Attention_Tensor->mem_len) = *(Linear->mem_len);

    (*Attention_Tensor)->shape = (int *)malloc(sizeof(int) * (*(Attention_Tensor->ndim)));
    for (int i = 0; i < *(*Attention_Tensor)->ndim; i++) {
        Attention_Tensor->shape[i] = Linear->shape[i];
    }

    // Allocate CUDA memory
    cudaMalloc((void **)&d_ndim, sizeof(int));
    cudaMalloc((void **)&d_mem_len, sizeof(int));
    cudaMalloc((void **)&d_shape, sizeof(int) * (*(Attention_Tensor->ndim)));
    cudaMalloc((void **)&d_fp16_tensor, sizeof(__half) * (*(Attention_Tensor->mem_len)));

    // Copy data to device
    cudaMemcpy(d_ndim, Attention_Tensor->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, Attention_Tensor->mem_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, Attention_Tensor->shape, sizeof(int) * (*(Attention_Tensor->ndim)), cudaMemcpyHostToDevice);

    // Assign device pointers
    Attention_Tensor->d_ndim = d_ndim;
    Attention_Tensor->d_mem_len = d_mem_len;
    Attention_Tensor->d_shape = d_shape;
    Attention_Tensor->d_fp16_tensor = d_fp16_tensor;

    return;
}

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V, Llama3Layer *L3_Layer, Tensor *X) {
    // Compute Queries
    kernel_compute_attention_tensors<<<1, 1>>>(
        Q, L3_Layer->self_attn_q_proj, X->d_fp16_tensor);

    cudaDeviceSynchronize();

    // Compute Keys
    kernel_compute_attention_tensors<<<1, 1>>>(
        K, L3_Layer->self_attn_k_proj, X->d_fp16_tensor);

    cudaDeviceSynchronize();

    // Compute Values
    kernel_compute_attention_tensors<<<1, 1>>>(
        V, L3_Layer->self_attn_v_proj, X->d_fp16_tensor);

    cudaDeviceSynchronize();
}

__global__ void kernel_compute_attention_tensors(Tensor *O, __half *Linear, __half *X) {
    // idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < (*(O->ndim)); i++) {
        printf("%d, \n", O->shape[i]);
    }
}