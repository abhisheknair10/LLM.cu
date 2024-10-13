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

__constant__ int EMBED_SIZE;

__device__ int d_NUM_TOKENS;
int h_NUM_TOKENS;

// Allocate global mem cache on device
float *create_gmemcache(size_t mem_len, size_t type_size) {
    float *d_gcache;

    cudaMalloc(&d_gcache, mem_len * type_size);

    return d_gcache;
}

void free_tensor_cuda(Tensor *t) {
    cudaFree(t->d_ndim);
    cudaFree(t->d_mem_len);
    cudaFree(t->d_shape);
    cudaFree(t->d_fp16_tensor);

    return;
}

// Print CUDA memory info
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

    return;
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
    int embed_size = 4096;
    cudaMemcpyToSymbol(EMBED_SIZE, &embed_size, sizeof(int));

    // Set NUM_TOKENS value in device memory
    h_NUM_TOKENS = h_tokens[0] - 1;
    cudaMemcpyToSymbol(d_NUM_TOKENS, &h_NUM_TOKENS, sizeof(int));
    free(h_tokens);

    tokens_to_embeddings(X, llama3_model, d_tokens);

    // Ahead Of Time memory allocations
    // Allocate once, use everywhere
    Tensor *PN_X = (Tensor *)malloc(sizeof(Tensor));
    _create_intermediary_prenorm_tensor_copy(PN_X, X);

    float *d_gcache = create_gmemcache(10000, sizeof(float));

    Tensor *Q = (Tensor *)malloc(sizeof(Tensor));
    Tensor *K = (Tensor *)malloc(sizeof(Tensor));
    Tensor *V = (Tensor *)malloc(sizeof(Tensor));
    _create_intermediary_attention_tensor(Q, llama3_model->layers[0]->self_attn_q_proj);
    _create_intermediary_attention_tensor(K, llama3_model->layers[0]->self_attn_k_proj);
    _create_intermediary_attention_tensor(V, llama3_model->layers[0]->self_attn_v_proj);

    // Run Inference
    for (int i = 0; i < llama3_model->n_layers; i++) {
        // Pre-attention normalization
        copy_fp16_tensor(PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->input_layernorm, X, d_gcache);

        // Attention computation
        compute_qkv_tensors(Q, K, V, llama3_model->layers[i], X);

        break;
    }

    printCudaMemoryInfo();

    free_tensor_cuda(PN_X);
    free_tensor_cuda(Q);
    free_tensor_cuda(K);
    free_tensor_cuda(V);
    cudaFree(d_gcache);

    return;
}

/* *************************** Convert Tokens to Embeddings *************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens) {
    // Order threads into blocks
    int total_threads = *(X->mem_len);
    int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kernel_tokens_to_embeddings<<<blocks, THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, llama3_model->embed_tokens->d_fp16_tensor, d_tokens);

    cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(X->d_fp16_tensor);
    // cudaDeviceSynchronize();

    return;
}

__global__ void kernel_tokens_to_embeddings(__half *X_tensor, __half *Embed, int *tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = d_NUM_TOKENS * EMBED_SIZE;

    if (idx >= total_elements) return;

    int token_idx = idx / EMBED_SIZE;
    int embed_idx = idx % EMBED_SIZE;

    X_tensor[(token_idx * EMBED_SIZE) + embed_idx] =
        Embed[(tokens[token_idx + 1] * EMBED_SIZE) + embed_idx];

    return;
}

/* ******************************* Layer Normalization ******************************* */
void _create_intermediary_prenorm_tensor_copy(Tensor *Y, Tensor *X) {
    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Y->ndim = (int *)malloc(sizeof(int));
    *(Y->ndim) = *(X->ndim);

    Y->mem_len = (int *)malloc(sizeof(int));
    *(Y->mem_len) = *(X->mem_len);

    Y->shape = (int *)malloc(sizeof(int) * (*(X->ndim)));
    for (int i = 0; i < (*(X->ndim)); i++) {
        Y->shape[i] = X->shape[i];
    }

    // Allocate CUDA memory
    cudaMalloc(&d_ndim, sizeof(int));
    cudaMalloc(&d_mem_len, sizeof(int));
    cudaMalloc(&d_shape, sizeof(int) * (*(Y->ndim)));
    cudaMalloc(&d_fp16_tensor, sizeof(__half) * (*(Y->mem_len)));

    // Copy data to device
    cudaMemcpy(d_ndim, Y->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, Y->mem_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, Y->shape, sizeof(int) * (*(Y->ndim)), cudaMemcpyHostToDevice);

    // Assign device pointers
    Y->d_ndim = d_ndim;
    Y->d_mem_len = d_mem_len;
    Y->d_shape = d_shape;
    Y->d_fp16_tensor = d_fp16_tensor;

    return;
}

void copy_fp16_tensor(Tensor *Y, Tensor *X) {
    cudaMemcpy(
        Y->d_fp16_tensor,
        X->d_fp16_tensor,
        sizeof(__half) * (*(Y->mem_len)),
        cudaMemcpyDeviceToDevice);

    return;
}

void compute_layer_norm(Tensor *RMSNorm, Tensor *X, float *d_gcache) {
    int blocks_x = 4096 / THREADS_PER_BLOCK;
    int blocks_y = h_NUM_TOKENS;

    dim3 blocks(blocks_x, blocks_y);
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(float);

    kernel_compute_rms_norm<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        X->d_fp16_tensor, RMSNorm->d_fp16_tensor, d_gcache);
    cudaDeviceSynchronize();

    kernel_compute_norm_tensor<<<blocks, THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, RMSNorm->d_fp16_tensor, d_gcache);
    cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(X->d_fp16_tensor);
    // cudaDeviceSynchronize();
}

__global__ void kernel_compute_rms_norm(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache) {
    extern __shared__ float shared_mem[];

    int token_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (embed_idx >= EMBED_SIZE) return;

    // Convert __half to float and square
    float x = __half2float(X_tensor[(token_idx * EMBED_SIZE) + embed_idx]);
    shared_mem[threadIdx.x] = x * x;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Store partial sums in d_gcache
    if (threadIdx.x == 0) {
        d_gcache[blockIdx.y * gridDim.x + blockIdx.x] = shared_mem[0];
    }
    __syncthreads();

    float rms = 0.0f;
    float eps = 1e-6f;

    // Compute the RMS value
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < gridDim.x; i++) {
            rms += d_gcache[blockIdx.y * gridDim.x + i];
        }
        rms = sqrtf((rms + eps) / (float)EMBED_SIZE);
        d_gcache[blockIdx.y] = rms;
    }

    return;
}

__global__ void kernel_compute_norm_tensor(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache) {
    int token_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (embed_idx >= EMBED_SIZE) return;

    // Normalize the input and write back
    float rms = d_gcache[blockIdx.y];
    float x = __half2float(X_tensor[(token_idx * EMBED_SIZE) + embed_idx]);
    float scale = __half2float(RMSNorm_tensor[embed_idx]);

    float res = (x / rms) * scale;
    X_tensor[(token_idx * EMBED_SIZE) + embed_idx] = __float2half(res);

    return;
}

/* ******************************* Attention Computation ******************************* */
void _create_intermediary_attention_tensor(Tensor *Attention_Tensor, Tensor *Linear) {
    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Attention_Tensor->ndim = (int *)malloc(sizeof(int));
    *(Attention_Tensor->ndim) = 2;

    Attention_Tensor->mem_len = (int *)malloc(sizeof(int));
    *(Attention_Tensor->mem_len) = Linear->shape[0] * h_NUM_TOKENS;

    Attention_Tensor->shape = (int *)malloc(sizeof(int) * 2);
    Attention_Tensor->shape[0] = h_NUM_TOKENS;
    Attention_Tensor->shape[1] = Linear->shape[0];

    // Allocate CUDA memory
    cudaMalloc(&d_ndim, sizeof(int));
    cudaMalloc(&d_mem_len, sizeof(int));
    cudaMalloc(&d_shape, sizeof(int) * 2);
    cudaMalloc(&d_fp16_tensor, sizeof(__half) * (*(Attention_Tensor->mem_len)));

    // Copy data to device
    cudaMemcpy(d_ndim, Attention_Tensor->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, Attention_Tensor->mem_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, Attention_Tensor->shape, sizeof(int) * 2, cudaMemcpyHostToDevice);

    // Assign device pointers
    Attention_Tensor->d_ndim = d_ndim;
    Attention_Tensor->d_mem_len = d_mem_len;
    Attention_Tensor->d_shape = d_shape;
    Attention_Tensor->d_fp16_tensor = d_fp16_tensor;

    return;
}

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V, Llama3Layer *L3_Layer, Tensor *X) {
    /*
    // Compute Queries
    kernel_compute_attention_tensors<<<4, 1024>>>(
        Q->d_fp16_tensor, Q->d_ndim, Q->d_shape,
        L3_Layer->self_attn_q_proj->d_fp16_tensor, L3_Layer->self_attn_q_proj->d_ndim, L3_Layer->self_attn_q_proj->d_shape,
        X->d_fp16_tensor, X->d_ndim, X->d_shape);
    cudaDeviceSynchronize();
    */

    return;
}

__global__ void kernel_compute_attention_tensors(
    __half *O_tensor, int *O_ndim, int *O_shape,
    __half *Linear_tensor, int *Linear_ndim, int *Linear_shape,
    __half *X_tensor, int *X_ndim, int *X_shape) {
    // Start of kernel

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = gridDim.x * blockDim.x;

    // [4096][1024, 4096]

    if (idx >= dim) return;

    __half sum = 0;
    for (int i = 0; i < EMBED_SIZE; i++) {
        sum += (X_tensor[i] * Linear_tensor[EMBED_SIZE * i]);
    }

    O_tensor[idx] = sum;

    return;
}