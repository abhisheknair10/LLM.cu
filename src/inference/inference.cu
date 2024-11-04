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

__device__ int EMBED_SIZE;

__device__ int d_NUM_TOKENS;
int h_NUM_TOKENS;

/* ************************************ HELPERS ************************************ */
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
        printf("Token %d embeddings:\n", token_idx);
        for (int i = 0; i < dim; i++) {
            printf("%f, ", __half2float(fp16_tensor[token_idx * dim + i]));
        }
        printf("\n");
        printf("\n\n");
    }

    return;
}
/*
__global__ void check_embedding(__half *fp16_tensor, int dim) {
    for (int token_idx = 0; token_idx < d_NUM_TOKENS; token_idx++) {
        printf("Token %d embeddings:\n", token_idx + 1);
        int max = 0;
        float curr_max = 0.0f;
        for (int i = 0; i < dim; i++) {
            float embedding = __half2float(fp16_tensor[token_idx * dim + i]);

            if (embedding > curr_max) {
                curr_max = embedding;
                max = i;
            }
        }
        printf("%d\n", max);
        printf("\n\n");
    }

    return;
}
*/
/* ************************************* Cache ************************************* */
// Allocate global mem cache on device
void *create_gmemcache(size_t mem_len, size_t type_size) {
    void *d_gcache;

    cudaMalloc(&d_gcache, mem_len * type_size);

    return d_gcache;
}

CudaCache *init_cache(Llama3 *llama3_model) {
    // Ahead Of Time memory allocations
    // Allocate once, use everywhere
    CudaCache *Cache = (CudaCache *)malloc(sizeof(CudaCache));

    // Allocate Memory --------------------------------------------------------
    Tensor *PN_X = _create_intermediary_prenorm_tensor_copy();

    Tensor *Q = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_q_proj);
    Tensor *K = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_k_proj);
    Tensor *V = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_v_proj);

    float *d_attention_score_cache = (float *)create_gmemcache(2048 * 2048, sizeof(float));

    __half *d_feedforward_cache_gate = (__half *)create_gmemcache(14336 * 2048, sizeof(__half));
    __half *d_feedforward_cache_up = (__half *)create_gmemcache(14336 * 2048, sizeof(__half));

    __half *next_token = (__half *)create_gmemcache(128256 * 2048, sizeof(__half));

    // Save pointers to Struct --------------------------------------------------------
    Cache->PN_X = PN_X;

    Cache->Q = Q;
    Cache->K = K;
    Cache->V = V;

    Cache->d_attention_score_cache = d_attention_score_cache;
    Cache->d_feedforward_cache_gate = d_feedforward_cache_gate;
    Cache->d_feedforward_cache_up = d_feedforward_cache_up;

    Cache->next_token = next_token;

    return Cache;
}

/* ********************************* Inference Code ********************************* */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache) {
    int embed_size = 4096;
    cudaMemcpyToSymbol(EMBED_SIZE, &embed_size, sizeof(int));

    // Set NUM_TOKENS value in device memory
    h_NUM_TOKENS = h_tokens[0] - 1;
    cudaMemcpyToSymbol(d_NUM_TOKENS, &h_NUM_TOKENS, sizeof(int));
    free(h_tokens);

    tokens_to_embeddings(X, llama3_model, d_tokens);

    for (int i = 0; i < llama3_model->n_layers; i++) {
        // Pre-attention normalization
        _deviceMemcpy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->input_layernorm, X);

        // Attention tensor computation
        compute_qkv_tensors(Cache->Q, Cache->K, Cache->V, llama3_model->layers[i], X);
        exit(1);
        // RoPE scaling
        rope_scaling(Cache->Q, Cache->K);

        // Attention computation
        compute_attention(X, Cache->Q, Cache->K, Cache->V, Cache);

        // Output computation
        compute_output(llama3_model->layers[i], X);

        // Add pre-normalized input
        add_norm(X, Cache->PN_X);

        // Post-attention normalization
        _deviceMemcpy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->post_attention_layernorm, X);

        // Feedforward
        compute_feedforward(X, llama3_model->layers[i], Cache);

        // Add pre-normalized input
        add_norm(X, Cache->PN_X);
    }

    compute_layer_norm(llama3_model->norm, X);
    compute_lm_head(llama3_model->lm_head, X, Cache);

    printCudaMemoryInfo();

    return;
}

/* ************************** Convert Tokens to Embeddings ************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens) {
    // Order threads into blocks
    int total_threads = *(X->mem_len);
    int blocks = (total_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    kernel_tokens_to_embeddings<<<blocks, MAX_THREADS_PER_BLOCK>>>(
        X->d_fp16_tensor, d_tokens, llama3_model->embed_tokens->d_fp16_tensor);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_tokens_to_embeddings(__half *X, int *tokens, __half *Embed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = d_NUM_TOKENS * EMBED_SIZE;

    if (idx >= total_elements) return;

    int token_idx = idx / EMBED_SIZE;
    int embed_idx = idx % EMBED_SIZE;

    X[(token_idx * EMBED_SIZE) + embed_idx] =
        Embed[(tokens[token_idx + 1] * EMBED_SIZE) + embed_idx];

    return;
}

/* ******************************* Layer Normalization ******************************* */
// Helpers
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
    Y->shape[1] = 4096;

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

void _deviceMemcpy_fp16_tensor(Tensor *Y, Tensor *X) {
    cudaMemcpy(
        Y->d_fp16_tensor,
        X->d_fp16_tensor,
        sizeof(__half) * (*(X->mem_len)),
        cudaMemcpyDeviceToDevice);

    return;
}

// Compute RMS Norm
void compute_layer_norm(Tensor *RMSNorm, Tensor *X) {
    dim3 block(1024);
    dim3 grid(h_NUM_TOKENS);

    kernel_compute_rms_norm<<<grid, block>>>(
        X->d_fp16_tensor, RMSNorm->d_fp16_tensor);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_compute_rms_norm(__half *X, __half *RMSNorm) {
    __shared__ float shared_mem[1024];

    int token_idx = blockIdx.x;
    int vw_embed_idx = threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (vw_embed_idx >= 1024) return;

    /*
        - Coalesced load into shared memory of 1024 window with vectorized retrieval
        - A 1024 thread block is used to retrieve 4096 elements. Each thread retrieves consecutive
            indicies. Instead of looping and having 4 separate memory access transactions for each
            window retrieval per thread, a singular call loading 4 __half's as 1 uint64_t allows for
            4 indicies to be retreived virtually as one data type.
    */
    c_half4 data = ((c_half4 *)X)[token_idx * 1024 + vw_embed_idx];
    shared_mem[vw_embed_idx] = __half2float(data.x) * __half2float(data.x) +
                               __half2float(data.y) * __half2float(data.y) +
                               __half2float(data.z) * __half2float(data.z) +
                               __half2float(data.w) * __half2float(data.w);
    __syncthreads();

    /*
        - Parallel reduction along y-axis (maximize warp usage without warp divergence)
        - For a 32 x 32 block dimension, the 1st warp will sum with the 16th warp and
            recursively reduce
    */
    for (int offset = 512; offset > 32; offset /= 2) {
        if (vw_embed_idx < offset) {
            shared_mem[vw_embed_idx] += shared_mem[offset + vw_embed_idx];
        }
        __syncthreads();
    }

    /*
        - Parallel reduction for 1 warp (divergent warp behavior) without using shared memory
        - Warp level primitive usage
        - Instead of utilizing shared memory to store intermediate reduction sums, inter-thread
            memory access enables faster reduction
        - For a given warp, the following will still not diverge with 0xffffff mask enabling the
            same instruction for every thread in the warp
        - Offset enables reduction to happen with left most indices lasting the longest. Least
            significant indices still perform addition but add no value to context
    */
    if (vw_embed_idx < 32) {
        __syncthreads();
        float val = shared_mem[vw_embed_idx];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
            __syncthreads();
        }
        if (vw_embed_idx == 0) shared_mem[0] = val;
    }

    /*
        - Load rms norm for tensor and perform normalization for 1024 window
        - Similar technique to when loading data from global memory
    */
    float rms = sqrtf((shared_mem[0] / 4096.0f) + 1e-05);
    __syncthreads();
    c_half4 norm_gain = ((c_half4 *)RMSNorm)[vw_embed_idx];

    // Perform RMS calculations and store
    data.x = __float2half(__half2float(data.x) * __half2float(norm_gain.x) / rms);
    data.y = __float2half(__half2float(data.y) * __half2float(norm_gain.y) / rms);
    data.z = __float2half(__half2float(data.z) * __half2float(norm_gain.z) / rms);
    data.w = __float2half(__half2float(data.w) * __half2float(norm_gain.w) / rms);

    ((c_half4 *)X)[token_idx * 1024 + vw_embed_idx] = data;

    return;
}

// Compute addition (skip connection)
void add_norm(Tensor *X, Tensor *PN_X) {
    dim3 block(32, 32, 1);
    dim3 grid(4, h_NUM_TOKENS);

    add_norm<<<grid, block>>>(
        X->d_fp16_tensor, PN_X->d_fp16_tensor);
    cudaDeviceSynchronize();

    return;
}

__global__ void add_norm(__half *X, __half *PN_X) {
    int token_idx = blockIdx.y;
    int embed_idx = blockIdx.x * 1024 +
                    threadIdx.y * blockDim.x +
                    threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (embed_idx >= 4096) return;

    int offset = token_idx * 4096 + embed_idx;
    X[offset] = __hadd(X[offset], PN_X[offset]);

    return;
}

/* ***************************** General Matrix Multiplication **************************** */
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
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
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
            value += X_shmem[threadIdx.y * TILE_SIZE + i] * T_shmem[i * TILE_SIZE + threadIdx.x];
        }
    }

    // Write the result to global memory
    if (row < m && col < n) {
        int O_idx = row * n + col;
        O[O_idx] = __float2half(value);
    }

    return;
}

/* ***************************** Attention Tensor Computation **************************** */
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

void compute_qkv_tensors(
    Tensor *Q, Tensor *K, Tensor *V,
    Llama3Layer *L3_Layer, Tensor *X) {
    // Declare common variables
    int TILE_SIZE = 32;
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid;

    // Query computation
    grid = dim3(
        (L3_Layer->self_attn_q_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        Q->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_q_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_q_proj->shape[0], 4096, TILE_SIZE);

    // Key computation
    grid = dim3(
        (L3_Layer->self_attn_k_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        K->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_k_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_k_proj->shape[0], 4096, TILE_SIZE);

    // Value computation
    grid = dim3(
        (L3_Layer->self_attn_v_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        V->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_v_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_v_proj->shape[0], 4096, TILE_SIZE);
    cudaDeviceSynchronize();

    check_embedding<<<1, 1>>>(Q->d_fp16_tensor, 4096);
    cudaDeviceSynchronize();

    return;
}

void compute_output(Llama3Layer *L3_Layer, Tensor *X) {
    // Declare common variables
    int TILE_SIZE = 32;
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid;

    // Output computation
    grid = dim3(
        (L3_Layer->self_attn_o_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        X->d_fp16_tensor, X->d_fp16_tensor, L3_Layer->self_attn_o_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->self_attn_o_proj->shape[0], 4096, TILE_SIZE);
    cudaDeviceSynchronize();

    return;
}

/* ************************* Rotary Positional Embedding (RoPE) ************************* */
void rope_scaling(Tensor *Q, Tensor *K) {
    dim3 block;
    dim3 grid;

    // RoPE on Q
    block = dim3(32, 32, 1);
    grid = dim3(2, h_NUM_TOKENS);
    kernel_rope_scaling<<<grid, block>>>(Q->d_fp16_tensor, 2048);

    // RoPE on K
    block = dim3(16, 16, 1);
    grid = dim3(2, h_NUM_TOKENS);
    kernel_rope_scaling<<<grid, block>>>(K->d_fp16_tensor, 512);

    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_rope_scaling(__half *tensor, int transformed_embed_size) {
    /*
        - For Q [tokens, 4096], there are 1024 threads per block with 2 blocks representing one
            transformed Q embedding
        - For K [tokens, 1024], there are 256 threads per block with 2 blocks representing one
            transformed K embedding
        - Window dim gives half the transformed tensor embedding size
        - Window idx gives local index
    */
    int token_idx = blockIdx.y;
    int window_idx = 2 * (blockIdx.x * blockDim.y * blockDim.x +
                          threadIdx.y * blockDim.x +
                          threadIdx.x);

    if (window_idx >= transformed_embed_size) return;
    if (token_idx >= d_NUM_TOKENS) return;

    // Each thread loads 2 __half (each 2 bytes), as one 4 byte value into half2 datatype
    __half2 h2_val = ((const __half2 *)tensor)[window_idx];

    const float scaling_factor = 500000.0f;
    float theta = token_idx / powf(scaling_factor, ((float)window_idx) / ((float)transformed_embed_size));
    float cos_comp = cosf(theta);
    float sin_comp = sinf(theta);

    // Access both values interpreted as 1 and rotate vector pair
    float even = __half2float(__low2half(h2_val));
    float odd = __half2float(__high2half(h2_val));

    float ret_even = (cos_comp * even) - (sin_comp * odd);
    float ret_odd = (sin_comp * even) + (cos_comp * odd);

    // Pack the two __half values into a single __half2
    __half h_ret_even = __float2half(ret_even);
    __half h_ret_odd = __float2half(ret_odd);
    __half2 h2_result = __halves2half2(h_ret_even, h_ret_odd);

    // Store rope encoded data back to tensor
    ((__half2 *)tensor)[window_idx] = h2_result;

    return;
}

/* **************************** Grouped Multi-Query Attention **************************** */
void compute_attention(Tensor *X, Tensor *Q, Tensor *K, Tensor *V, CudaCache *Cache) {
    // Attention score computation
    int TILE_SIZE = 32;
    int nheads = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE,
        nheads);

    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    kernel_compute_masked_gmq_attention_scores_tiled_matmul<<<grid, block, shared_mem_size>>>(
        Cache->d_attention_score_cache, Q->d_fp16_tensor, K->d_fp16_tensor,
        h_NUM_TOKENS, h_NUM_TOKENS, 128, TILE_SIZE, nheads);
    cudaDeviceSynchronize();

    // Masking and softmax
    block = dim3(MAX_THREADS_PER_BLOCK);
    grid = dim3(h_NUM_TOKENS, nheads);

    shared_mem_size = (2048 + 1024) * sizeof(float);
    kernel_masking_softmax<<<grid, block, shared_mem_size>>>(
        Cache->d_attention_score_cache, 1, 1);
    cudaDeviceSynchronize();

    // Resolution of attention scores
    block = dim3(TILE_SIZE, TILE_SIZE);
    grid = dim3(
        (128 + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE,
        nheads);

    kernel_compute_resolved_value_from_attention_score_tiled_matmul<<<grid, block, shared_mem_size>>>(
        X->d_fp16_tensor, Cache->d_attention_score_cache, V->d_fp16_tensor,
        h_NUM_TOKENS, h_NUM_TOKENS, 128, nheads, TILE_SIZE);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_compute_masked_gmq_attention_scores_tiled_matmul(
    float *attention_scores, __half *Q, __half *K,
    int m, int n, int k, int TILE_SIZE, int nheads) {
    /*
        - Each head operates independently of other heads.
        - `m` represents the independent dimension of the Q matrix (number of tokens).
        - `n` represents the independent dimension of the K matrix (number of tokens).
        - `k` represents the common dimension (embedding dimension for each head).
    */

    extern __shared__ float shared_mem[];

    float *Q_shmem = shared_mem;
    float *K_shmem = shared_mem + (TILE_SIZE * TILE_SIZE);

    int head_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Loop over tiles
    float value = 0.0f;
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int embedding_idx = t * TILE_SIZE + threadIdx.x;

        // Load tile of Q into shared memory
        if (row < m && embedding_idx < k) {
            int Q_idx = row * nheads * k + head_idx * k + embedding_idx;
            Q_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = __half2float(Q[Q_idx]);
        } else {
            Q_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Load tile of K into shared memory (transposed)
        if (col < n && embedding_idx < k) {
            int K_idx = (col * nheads * k / 4) + (head_idx * k / 4) + embedding_idx;
            K_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = __half2float(K[K_idx]);
        } else {
            K_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Compute partial sums
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += Q_shmem[threadIdx.y * TILE_SIZE + i] * K_shmem[i * TILE_SIZE + threadIdx.x];
        }
    }

    // Write the result to shared memory
    if (row < m && col < n) {
        int O_idx = row * nheads * n + head_idx * n + col;
        attention_scores[O_idx] = value / sqrtf((float)k);
    }

    return;
}

__global__ void kernel_masking_softmax(float *attention_scores, int causal_mask, int softmax) {
    extern __shared__ float shared_mem[];

    float *buffer = shared_mem + 2048;

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int idx;
    float exp_sum = 0.0f;
    for (int i = 0; i < 2; i++) {
        idx = i * blockDim.x + threadIdx.x;

        if (idx >= d_NUM_TOKENS) {
            shared_mem[idx] = 0.0f;
            continue;
        }

        if (causal_mask) {
            if (idx <= token_idx) {
                shared_mem[idx] = attention_scores[blockIdx.y * 2048 + head_idx * 32 + idx];
            }
        } else {
            shared_mem[idx] = -INFINITY;
        }

        exp_sum += expf(shared_mem[idx]);
    }
    __syncthreads();

    buffer[threadIdx.x] = exp_sum;
    if (softmax) {
        for (int offset = 512; offset > 32; offset /= 2) {
            if (threadIdx.x < offset) {
                buffer[threadIdx.x] += buffer[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            float val = buffer[threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (threadIdx.x == 0) buffer[0] = val;
        }
        __syncthreads();

        float softmax_den = buffer[0];
        for (int i = 0; i < 2; i++) {
            idx = i * blockDim.x + threadIdx.x;
            attention_scores[blockIdx.y * 2048 + head_idx * 32 + idx] = expf(shared_mem[idx]) / softmax_den;
            __syncthreads();
        }
    }

    return;
}

__global__ void kernel_compute_resolved_value_from_attention_score_tiled_matmul(
    __half *output, float *attention_scores, __half *V,
    int m, int k, int d_head, int nheads, int TILE_SIZE) {
    // Kernel start
    //
    extern __shared__ float shared_mem[];
    float *attention_shmem = shared_mem;
    float *V_shmem = shared_mem + TILE_SIZE * TILE_SIZE;

    int head_idx = blockIdx.z;
    int V_head_idx = head_idx / 4;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_idx = t * TILE_SIZE + threadIdx.x;  // K dimension index

        // Load attention_scores into shared memory
        if (row < m && k_idx < k) {
            int attn_idx = head_idx * m * k + row * k + k_idx;
            attention_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = attention_scores[attn_idx];
        } else {
            attention_shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Load V into shared memory
        int V_row = k_idx;
        int V_col = col;
        if (V_row < k && V_col < d_head) {
            int V_idx = V_head_idx * k * d_head + V_row * d_head + V_col;
            V_shmem[threadIdx.x * TILE_SIZE + threadIdx.y] = __half2float(V[V_idx]);
        } else {
            V_shmem[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += attention_shmem[threadIdx.y * TILE_SIZE + i] * V_shmem[i * TILE_SIZE + threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output tensor
    if (row < m && col < d_head) {
        int output_idx = head_idx * m * d_head + row * d_head + col;
        output[output_idx] = __float2half(value);
    }
}

/* ********************************* Feed Forward Network ********************************* */
void compute_feedforward(Tensor *X, Llama3Layer *L3_Layer, CudaCache *Cache) {
    // Declare common variables
    int TILE_SIZE = 32;
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid;

    // Gate projection computation
    grid = dim3(
        (L3_Layer->mlp_gate_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        Cache->d_feedforward_cache_gate, X->d_fp16_tensor, L3_Layer->mlp_gate_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->mlp_gate_proj->shape[0], 4096, TILE_SIZE);

    // Up projection computation
    grid = dim3(
        (L3_Layer->mlp_up_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        Cache->d_feedforward_cache_up, X->d_fp16_tensor, L3_Layer->mlp_up_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->mlp_up_proj->shape[0], 4096, TILE_SIZE);
    cudaDeviceSynchronize();

    // Swiglu Activation
    grid = dim3(
        (L3_Layer->mlp_up_proj->shape[0] + 1024 - 1) / 1024,
        h_NUM_TOKENS);

    kernel_compute_swiglu<<<grid, block>>>(
        Cache->d_feedforward_cache_up, Cache->d_feedforward_cache_gate, Cache->d_feedforward_cache_up,
        L3_Layer->mlp_up_proj->shape[0]);
    cudaDeviceSynchronize();

    // Final output feedforward output computation
    grid = dim3(
        (L3_Layer->mlp_down_proj->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        X->d_fp16_tensor, Cache->d_feedforward_cache_up, L3_Layer->mlp_down_proj->d_fp16_tensor,
        h_NUM_TOKENS, L3_Layer->mlp_down_proj->shape[0], L3_Layer->mlp_up_proj->shape[0], TILE_SIZE);
    cudaDeviceSynchronize();

    return;
}

__device__ float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

__global__ void kernel_compute_swiglu(__half *output, __half *gate, __half *up, int embed_dim) {
    int flat_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int embed_idx = blockIdx.x * blockDim.y * blockDim.x + flat_thread_idx;
    int token_idx = blockIdx.y;

    float gate_val = __half2float(gate[token_idx * embed_dim + embed_idx]);
    float up_val = __half2float(up[token_idx * embed_dim + embed_idx]);

    output[token_idx * embed_dim + embed_idx] = __float2half(
        sigmoid(gate_val) * up_val);

    return;
}

/* ********************************* Language Model Head ********************************* */
void compute_lm_head(Tensor *LM_Head, Tensor *X, CudaCache *Cache) {
    // Declare common variables
    int TILE_SIZE = 32;
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid;

    // Query computation
    grid = dim3(
        (LM_Head->shape[0] + TILE_SIZE - 1) / TILE_SIZE,
        (h_NUM_TOKENS + TILE_SIZE - 1) / TILE_SIZE);

    kernel_standard_tiled_gemm<<<grid, block, shared_mem_size>>>(
        Cache->next_token, X->d_fp16_tensor, LM_Head->d_fp16_tensor,
        h_NUM_TOKENS, LM_Head->shape[0], 4096, TILE_SIZE);
    cudaDeviceSynchronize();

    check_embedding<<<1, 1>>>(Cache->next_token, 128256);
    cudaDeviceSynchronize();

    return;
}
