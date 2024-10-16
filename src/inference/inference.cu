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

const int MAX_THREADS_PER_BLOCK = 1024;

__constant__ int EMBED_SIZE;

__device__ int d_NUM_TOKENS;
int h_NUM_TOKENS;

/* ******************************** HELPERS ******************************** */
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
__global__ void check_embedding(__half *fp16_tensor, int dim) {
    for (int token_idx = 0; token_idx < d_NUM_TOKENS; token_idx++) {
        printf("Token %d embeddings:\n", token_idx + 1);
        for (int i = 0; i < dim; i++) {
            float embedding = __half2float(fp16_tensor[token_idx * dim + i]);
            printf("%f ", embedding);
        }
        printf("\n\n\n\n\n");
    }

    return;
}

/* ******************************** Cache ******************************** */
CudaCache *init_cache(Llama3 *llama3_model) {
    // Ahead Of Time memory allocations
    // Allocate once, use everywhere
    CudaCache *Cache = (CudaCache *)malloc(sizeof(CudaCache));

    // Allocate Memory --------------------------------------------------------
    Tensor *PN_X = _create_intermediary_prenorm_tensor_copy();

    float *d_gnorm_cache = create_gmemcache(200000000, sizeof(float));
    float *d_attq_cache = create_gmemcache(50000000, sizeof(float));
    float *d_attk_cache = create_gmemcache(10000000, sizeof(float));
    float *d_attv_cache = create_gmemcache(10000000, sizeof(float));

    Tensor *Q = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_q_proj);
    Tensor *K = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_k_proj);
    Tensor *V = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_v_proj);

    // Save pointers to Struct --------------------------------------------------------
    Cache->PN_X = PN_X;

    Cache->d_gnorm_cache = d_gnorm_cache;
    Cache->d_attq_cache = d_attq_cache;
    Cache->d_attk_cache = d_attk_cache;
    Cache->d_attv_cache = d_attv_cache;

    Cache->Q = Q;
    Cache->K = K;
    Cache->V = V;

    return Cache;
}

/* ******************************** Inference Code ******************************** */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache) {
    int embed_size = 4096;
    cudaMemcpyToSymbol(EMBED_SIZE, &embed_size, sizeof(int));

    // Set NUM_TOKENS value in device memory
    h_NUM_TOKENS = h_tokens[0] - 1;
    cudaMemcpyToSymbol(d_NUM_TOKENS, &h_NUM_TOKENS, sizeof(int));
    free(h_tokens);

    tokens_to_embeddings(X, llama3_model, d_tokens);

    // Run Inference
    for (int i = 0; i < llama3_model->n_layers; i++) {
        // Pre-attention normalization
        copy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->input_layernorm, X, Cache->d_gnorm_cache);

        // Attention computation
        compute_qkv_tensors(Cache->Q, Cache->K, Cache->V, llama3_model->layers[i], X, Cache);

        // RoPE scaling
        rope_scaling(Cache->Q, Cache->K);

        break;
    }

    printCudaMemoryInfo();

    return;
}

/* *************************** Convert Tokens to Embeddings *************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens) {
    // Order threads into blocks
    int total_threads = *(X->mem_len);
    int blocks = (total_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    kernel_tokens_to_embeddings<<<blocks, MAX_THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, llama3_model->embed_tokens->d_fp16_tensor, d_tokens);

    cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(X->d_fp16_tensor, 4096);
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
Tensor *_create_intermediary_prenorm_tensor_copy() {
    Tensor *Y = (Tensor *)malloc(sizeof(Tensor));

    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Y->ndim = (int *)malloc(sizeof(int));
    *(Y->ndim) = 2;

    Y->mem_len = (int *)malloc(sizeof(int));
    *(Y->mem_len) = 4096 * 2048;

    Y->shape = (int *)malloc(sizeof(int) * 2);
    Y->shape[0] = 2048;
    Y->shape[0] = 4096;

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

    return Y;
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
    int blocks_x = 4096 / MAX_THREADS_PER_BLOCK;
    int blocks_y = h_NUM_TOKENS;

    dim3 blocks(blocks_x, blocks_y);
    size_t shared_mem_size = MAX_THREADS_PER_BLOCK * sizeof(float);

    kernel_compute_rms_norm<<<blocks, MAX_THREADS_PER_BLOCK, shared_mem_size>>>(
        X->d_fp16_tensor, RMSNorm->d_fp16_tensor, d_gcache);
    cudaDeviceSynchronize();

    kernel_compute_norm_tensor<<<blocks, MAX_THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, RMSNorm->d_fp16_tensor, d_gcache);
    cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(X->d_fp16_tensor, 4096);
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
Tensor *_create_intermediary_attention_tensor(Tensor *Linear) {
    Tensor *Attention_Tensor = (Tensor *)malloc(sizeof(Tensor));

    int *d_ndim;
    int *d_mem_len;
    int *d_shape;
    __half *d_fp16_tensor;

    Attention_Tensor->ndim = (int *)malloc(sizeof(int));
    *(Attention_Tensor->ndim) = 2;

    Attention_Tensor->mem_len = (int *)malloc(sizeof(int));
    *(Attention_Tensor->mem_len) = Linear->shape[0] * 2048;

    Attention_Tensor->shape = (int *)malloc(sizeof(int) * 2);
    Attention_Tensor->shape[0] = 2048;
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

    return Attention_Tensor;
}

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V,
                         Llama3Layer *L3_Layer, Tensor *X,
                         CudaCache *Cache) {
    // -------- Compute intermediate matmul in cache --------
    _abstract_intermediate_attensor_kernel_call(L3_Layer->self_attn_q_proj, X, Cache->d_attq_cache);
    _abstract_intermediate_attensor_kernel_call(L3_Layer->self_attn_k_proj, X, Cache->d_attk_cache);
    _abstract_intermediate_attensor_kernel_call(L3_Layer->self_attn_v_proj, X, Cache->d_attv_cache);

    cudaDeviceSynchronize();

    // -------- Compute full matmul in output tensors --------
    _abstract_full_attensor_kernel_call(Q, L3_Layer->self_attn_q_proj, Cache->d_attq_cache);
    _abstract_full_attensor_kernel_call(K, L3_Layer->self_attn_k_proj, Cache->d_attk_cache);
    _abstract_full_attensor_kernel_call(V, L3_Layer->self_attn_v_proj, Cache->d_attv_cache);

    cudaDeviceSynchronize();

    // ------------------------- Checks -------------------------
    // check_embedding<<<1, 1>>>(Q->d_fp16_tensor, 4096);
    // cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(K->d_fp16_tensor, 1024);
    // cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(V->d_fp16_tensor, 1024);
    // cudaDeviceSynchronize();

    return;
}

void _abstract_intermediate_attensor_kernel_call(Tensor *Proj_Layer, Tensor *X, float *d_gcache) {
    // Function start
    //
    int blockx, blocky, blockz;
    dim3 blocks;

    blockx = 4096 / MAX_THREADS_PER_BLOCK;
    blocky = Proj_Layer->shape[0];
    blockz = h_NUM_TOKENS;

    blocks = dim3(blockx, blocky, blockz);

    size_t shared_mem_size = MAX_THREADS_PER_BLOCK * sizeof(float);

    kernel_compute_intermediate_attention_matmul<<<blocks, MAX_THREADS_PER_BLOCK, shared_mem_size>>>(
        Proj_Layer->d_fp16_tensor, Proj_Layer->d_shape, X->d_fp16_tensor, d_gcache);
}

__global__ void kernel_compute_intermediate_attention_matmul(
    __half *Linear_tensor, int *Linear_shape,
    __half *X_tensor, float *d_gcache) {
    extern __shared__ float shared_mem[];

    int total_blocks_x = (EMBED_SIZE + blockDim.x - 1) / blockDim.x;

    int token_idx = blockIdx.z;
    int fcoord_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (fcoord_idx >= Linear_shape[0]) return;
    if (embed_idx >= EMBED_SIZE) return;

    float x = __half2float(X_tensor[token_idx * EMBED_SIZE + embed_idx]);
    float f = __half2float(Linear_tensor[fcoord_idx * EMBED_SIZE + embed_idx]);
    shared_mem[threadIdx.x] = x * f;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int cache_idx = token_idx * Linear_shape[0] * total_blocks_x +
                        fcoord_idx * total_blocks_x +
                        blockIdx.x;
        d_gcache[cache_idx] = shared_mem[0];
    }
}

void _abstract_full_attensor_kernel_call(Tensor *Attention_Tensor, Tensor *Proj_Layer, float *d_gcache) {
    // Function start
    //
    int blockx, blocky;
    dim3 blocks;

    blockx = Proj_Layer->shape[0] / MAX_THREADS_PER_BLOCK;
    blocky = h_NUM_TOKENS;
    blocks = dim3(blockx, blocky);

    kernel_compute_full_attention_tensors<<<blocks, MAX_THREADS_PER_BLOCK>>>(
        Attention_Tensor->d_fp16_tensor, Proj_Layer->d_shape, d_gcache);
}

__global__ void kernel_compute_full_attention_tensors(
    __half *O_tensor, int *Linear_shape, float *d_gcache) {
    int total_blocks_x = (EMBED_SIZE + blockDim.x - 1) / blockDim.x;

    int token_idx = blockIdx.y;
    int fcoord_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (fcoord_idx >= Linear_shape[0]) return;

    float sum = 0.0f;
    for (int i = 0; i < total_blocks_x; i++) {
        int cache_idx = token_idx * Linear_shape[0] * total_blocks_x +
                        fcoord_idx * total_blocks_x +
                        i;
        sum += d_gcache[cache_idx];
    }

    O_tensor[token_idx * Linear_shape[0] + fcoord_idx] = __float2half(sum);
}

/* ************************ Rotary Positional Embedding (RoPE) ************************ */
void rope_scaling(Tensor *Q, Tensor *K) {
    dim3 blocks;

    // RoPE on Q
    blocks = dim3(Q->shape[1] / (MAX_THREADS_PER_BLOCK * 2), h_NUM_TOKENS);
    kernel_rope_scaling<<<blocks, MAX_THREADS_PER_BLOCK>>>(Q->d_fp16_tensor);

    // RoPE on K
    blocks = dim3(K->shape[1] / (MAX_THREADS_PER_BLOCK * 2), h_NUM_TOKENS);
    kernel_rope_scaling<<<blocks, MAX_THREADS_PER_BLOCK>>>(K->d_fp16_tensor);

    cudaDeviceSynchronize();

    // ------------------------- Checks -------------------------
    // check_embedding<<<1, 1>>>(Q->d_fp16_tensor, 4096);
    // cudaDeviceSynchronize();

    // check_embedding<<<1, 1>>>(K->d_fp16_tensor, 1024);
    // cudaDeviceSynchronize();

    return;
}

__global__ void kernel_rope_scaling(__half *tensor) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int embed_idx = idx * 2;
    int token_idx = blockIdx.y;

    if (embed_idx >= EMBED_SIZE) return;
    if (token_idx >= d_NUM_TOKENS) return;

    int scaling_factor = 10000;
    float even = __half2float(tensor[token_idx * EMBED_SIZE + embed_idx]);
    float odd = __half2float(tensor[token_idx * EMBED_SIZE + embed_idx + 1]);

    float theta = token_idx * pow(scaling_factor, -2 * idx / EMBED_SIZE);
    float cos_comp = cos(theta);
    float sin_comp = sin(theta);

    tensor[token_idx * EMBED_SIZE + embed_idx] = __float2half((cos_comp * even) + (-1.0 * sin_comp * odd));
    tensor[token_idx * EMBED_SIZE + embed_idx + 1] = __float2half((sin_comp * even) + (cos_comp * odd));

    return;
}

/* ***************************** Multi-Head Attention ***************************** */