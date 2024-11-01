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
__global__ void check_embedding(float *fp16_tensor, int dim) {
    for (int token_idx = 0; token_idx < d_NUM_TOKENS; token_idx++) {
        printf("Token %d embeddings:\n", token_idx + 1);
        int max = 0;
        float curr_max = 0.0f;
        for (int i = 0; i < dim; i++) {
            float embedding = fp16_tensor[token_idx * dim + i];

            if (embedding > curr_max) {
                curr_max = embedding;
                max = i;
            }
        }
        printf("%d\n", max);
        printf("\n\n\n\n\n");
    }

    return;
}

/* ************************************* Cache ************************************* */
// Allocate global mem cache on device
float *create_gmemcache(size_t mem_len, size_t type_size) {
    float *d_gcache;

    cudaMalloc(&d_gcache, mem_len * type_size);

    return d_gcache;
}

CudaCache *init_cache(Llama3 *llama3_model) {
    // Ahead Of Time memory allocations
    // Allocate once, use everywhere
    CudaCache *Cache = (CudaCache *)malloc(sizeof(CudaCache));

    // Allocate Memory --------------------------------------------------------
    Tensor *PN_X = _create_intermediary_prenorm_tensor_copy();

    float *d_gnorm_cache = create_gmemcache(2048 * 1024, sizeof(float));
    float *d_attq_cache = create_gmemcache(2048 * 4096, sizeof(float));
    float *d_attk_cache = create_gmemcache(2048 * 1024, sizeof(float));
    float *d_attv_cache = create_gmemcache(2048 * 1024, sizeof(float));

    Tensor *Q = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_q_proj);
    Tensor *K = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_k_proj);
    Tensor *V = _create_intermediary_attention_tensor(llama3_model->layers[0]->self_attn_v_proj);

    float *d_attention_score_cache = create_gmemcache(2048 * 2048, sizeof(float));
    float *d_feedforward_cache = create_gmemcache(14336 * 2048, sizeof(float));

    float *next_token = create_gmemcache(128256 * 2048, sizeof(float));

    // Save pointers to Struct --------------------------------------------------------
    Cache->d_gnorm_cache = d_gnorm_cache;
    Cache->d_attq_cache = d_attq_cache;
    Cache->d_attk_cache = d_attk_cache;
    Cache->d_attv_cache = d_attv_cache;

    Cache->PN_X = PN_X;

    Cache->Q = Q;
    Cache->K = K;
    Cache->V = V;

    Cache->d_attention_score_cache = d_attention_score_cache;
    Cache->d_feedforward_cache = d_feedforward_cache;

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
        deviceMemcpy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->input_layernorm, X);

        //======================== COMPLETED AND CHECKED ========================
        // Attention tensor computation
        compute_qkv_tensors(Cache->Q, Cache->K, Cache->V, llama3_model->layers[i], X, Cache);

        // RoPE scaling
        rope_scaling(Cache->Q, Cache->K);

        // Attention computation
        compute_attention(X, Cache->Q, Cache->K, Cache->V, Cache);

        // Output computation
        compute_output(llama3_model->layers[i], X, Cache);

        // Add pre-normalized input
        add_norm(X, Cache->PN_X);

        // Post-attention normalization
        deviceMemcpy_fp16_tensor(Cache->PN_X, X);
        compute_layer_norm(llama3_model->layers[i]->post_attention_layernorm, X);

        // Feedforward
        compute_feedforward(X, llama3_model->layers[i], Cache);

        // Add pre-normalized input
        add_norm(X, Cache->PN_X);
        CHECK_CUDA_ERROR();
        break;
    }

    compute_layer_norm(llama3_model->norm, X);

    compute_lm_head(X, llama3_model->lm_head, Cache);

    CHECK_CUDA_ERROR();

    printCudaMemoryInfo();

    return;
}

/* ************************** Convert Tokens to Embeddings ************************** */
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

__global__ void kernel_tokens_to_embeddings(__half *X, __half *Embed, int *tokens) {
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

void deviceMemcpy_fp16_tensor(Tensor *Y, Tensor *X) {
    cudaMemcpy(
        Y->d_fp16_tensor,
        X->d_fp16_tensor,
        sizeof(__half) * (*(X->mem_len)),
        cudaMemcpyDeviceToDevice);

    return;
}

void compute_layer_norm(Tensor *RMSNorm, Tensor *X) {
    dim3 block(32, 32, 1);
    dim3 grid(h_NUM_TOKENS);

    kernel_compute_rms_norm<<<grid, block>>>(
        RMSNorm->d_fp16_tensor, X->d_fp16_tensor);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_compute_rms_norm(__half *RMSNorm, __half *X) {
    __shared__ float shared_mem[1024];

    int token_idx = blockIdx.x;
    int vw_embed_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (vw_embed_idx >= 1024) return;

    /*
        - Coalesced load into shared memory of 1024 window
        - For a 4096 1D vector, 1024 contiguous elements are loaded into shared memory 4 times
        - Each loop adds the square of the virtual embedding value
    */
    int tensor_offset;
    float _sum_sq = 0.0f;
    for (int i = 0; i < 4; i++) {
        tensor_offset = (token_idx * 4096) + (i * 1024 + vw_embed_idx);
        float tmp = __half2float(X[tensor_offset]);

        _sum_sq += (tmp * tmp);
    }

    shared_mem[vw_embed_idx] = _sum_sq;
    __syncthreads();

    /*
        - Parallel reduction along y-axis (maximize warp usage without warp divergence)
        - For a 32 x 32 block dimension, the 1st warp will sum with the 16th warp and
            recursively reduce
    */
    for (int offset = 512; offset > 32; offset /= 2) {
        if (vw_embed_idx < offset) {
            shared_mem[vw_embed_idx] += shared_mem[vw_embed_idx + offset];
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
        float val = shared_mem[vw_embed_idx];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (vw_embed_idx == 0) shared_mem[0] = val;
    }

    /*
        - Load rms norm for tensor and perform normalization for 1024 window
        - Similar technique to when loading data from global memory
    */
    float rms = sqrtf(shared_mem[0] / 4096);
    __syncthreads();
    for (int i = 0; i < 4; i++) {
        tensor_offset = (token_idx * 4096) + (i * 1024 + vw_embed_idx);
        X[tensor_offset] = __float2half(
            __half2float(X[tensor_offset]) *
            __half2float(RMSNorm[tensor_offset]) / rms);
    }

    return;
}

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
    X[offset] = X[offset] + PN_X[offset];

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

void compute_output(Llama3Layer *L3_Layer, Tensor *X, CudaCache *Cache) {
    _abstract_intermediate_attensor_kernel_call(L3_Layer->self_attn_o_proj, X, Cache->d_attq_cache);
    cudaDeviceSynchronize();

    _abstract_full_attensor_kernel_call(X, L3_Layer->self_attn_o_proj, Cache->d_attq_cache);
    cudaDeviceSynchronize();

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

    return;
}

__global__ void kernel_compute_intermediate_attention_matmul(
    __half *Linear_tensor, int *Linear_shape,
    __half *X, float *d_gcache) {
    extern __shared__ float shared_mem[];

    int total_blocks_x = (EMBED_SIZE + blockDim.x - 1) / blockDim.x;

    int token_idx = blockIdx.z;
    int fcoord_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS) return;
    if (fcoord_idx >= Linear_shape[0]) return;
    if (embed_idx >= EMBED_SIZE) return;

    float x = __half2float(X[token_idx * EMBED_SIZE + embed_idx]);
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

    return;
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
    int window_dim = gridDim.x * blockDim.y * blockDim.x;
    int window_idx = 2 * (blockIdx.x * blockDim.y * blockDim.x +
                          threadIdx.y * blockDim.x +
                          threadIdx.x);

    if (window_idx >= transformed_embed_size) return;
    if (token_idx >= d_NUM_TOKENS) return;

    // Each thread loads 2 __half (each 2 bytes), as one 4 byte value into half2 datatype
    __half2 h2_val = ((const __half2 *)tensor)[window_idx];

    const int scaling_factor = 500000;
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
    int nheads = 32;
    int nkheads = 8;
    int num_tokens = h_NUM_TOKENS;
    int head_dim = 128;

    // Kernel 1: Compute attention scores and apply softmax
    int blockSize1 = head_dim;
    int gridSize1_x = (num_tokens + blockSize1 - 1) / blockSize1;
    dim3 grid1(gridSize1_x, num_tokens, nheads);
    size_t shared_mem_size1 = blockSize1 * sizeof(float);

    kernel_compute_attention_scores_softmax<<<grid1, blockSize1, shared_mem_size1>>>(
        Cache->d_attention_score_cache, Q->d_fp16_tensor, K->d_fp16_tensor,
        num_tokens, nheads, nkheads, head_dim, 1, 1);
    cudaDeviceSynchronize();

    // Kernel 2: Multiply attention weights with V
    int blockSize2 = head_dim;
    dim3 grid2(1, num_tokens, nheads);
    size_t shared_mem_size2 = 0;

    kernel_compute_attention_output<<<grid2, blockSize2, shared_mem_size2>>>(
        X->d_fp16_tensor, Cache->d_attention_score_cache, V->d_fp16_tensor,
        num_tokens, nheads, nkheads, head_dim);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_compute_attention_scores_softmax(
    float *attention_scores, __half *Q, __half *K,
    int num_tokens, int nheads, int nkheads, int head_dim,
    uint8_t scaling, uint8_t automask) {
    // Kernel Start
    //
    extern __shared__ float shared_mem[];

    int h = blockIdx.z;
    int i = blockIdx.y;
    int j_start = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (i >= num_tokens || h >= nheads)
        return;

    int kv_head = h / (nheads / nkheads);

    // Compute attention scores for a chunk of keys
    for (int j = j_start + tid; j < num_tokens; j += blockDim.x) {
        if (j >= num_tokens)
            continue;

        // Dot product over head_dim
        float dot_product = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int Q_idx = i * nheads * head_dim + h * head_dim + d;
            int K_idx = j * nkheads * head_dim + kv_head * head_dim + d;

            float q_val = __half2float(Q[Q_idx]);
            float k_val = __half2float(K[K_idx]);
            dot_product += q_val * k_val;
        }

        // Apply scaling
        if (scaling) {
            dot_product /= sqrtf((float)head_dim);
        }

        // Apply mask
        if (automask && j > i) {
            dot_product = -1e9f;
        }

        // Store the raw attention score
        int score_idx = h * num_tokens * num_tokens + i * num_tokens + j;
        shared_mem[tid] = dot_product;

        __syncthreads();

        // Compute max for numerical stability
        float max_score = -1e9f;
        for (int k = 0; k < blockDim.x && (j_start + k) < num_tokens; ++k) {
            max_score = fmaxf(max_score, shared_mem[k]);
        }

        __syncthreads();

        // Subtract max and exponentiate
        float sum_exp = 0.0f;
        float exp_score = expf(shared_mem[tid] - max_score);
        shared_mem[tid] = exp_score;

        __syncthreads();

        // Compute sum of exponentials
        for (int k = 0; k < blockDim.x && (j_start + k) < num_tokens; ++k) {
            sum_exp += shared_mem[k];
        }

        __syncthreads();

        // Normalize to get softmax probabilities
        float attention_weight = exp_score / (sum_exp + 1e-6f);

        attention_scores[score_idx] = attention_weight;
    }

    return;
}

__global__ void kernel_compute_attention_output(
    __half *output, float *attention_scores, __half *V,
    int num_tokens, int nheads, int nkheads, int head_dim) {
    // Kernel start
    //
    extern __shared__ float shared_V[];

    int h = blockIdx.z;
    int i = blockIdx.y;
    int d = threadIdx.x;

    if (i >= num_tokens) return;
    if (h >= nheads) return;
    if (d >= head_dim) return;

    int kv_head = h / (nheads / nkheads);

    float output_val = 0.0f;

    // Loop over all keys to compute the weighted sum
    for (int j = 0; j < num_tokens; ++j) {
        int score_idx = h * num_tokens * num_tokens + i * num_tokens + j;
        float attn_weight = attention_scores[score_idx];

        int V_idx = j * nkheads * head_dim + kv_head * head_dim + d;
        float v_val = __half2float(V[V_idx]);

        output_val += attn_weight * v_val;
    }

    int output_idx = i * nheads * head_dim + h * head_dim + d;
    output[output_idx] = __float2half(output_val);

    return;
}

/* ********************************* Feed Forward Network ********************************* */
void compute_feedforward(Tensor *X, Llama3Layer *L3_Layer, CudaCache *Cache) {
    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((4096 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, h_NUM_TOKENS);

    size_t shared_mem_size = 2 * MAX_THREADS_PER_BLOCK * sizeof(float);

    // Gate and up projection
    kernel_gate_up_proj<<<gridDim, blockDim, shared_mem_size>>>(
        Cache->d_feedforward_cache, L3_Layer->mlp_up_proj->d_fp16_tensor,
        L3_Layer->mlp_gate_proj->d_fp16_tensor, X->d_fp16_tensor);
    cudaDeviceSynchronize();

    // Down projection
    int down_proj_out_dim = L3_Layer->mlp_down_proj->shape[0];
    dim3 blockDim_down(MAX_THREADS_PER_BLOCK);
    dim3 gridDim_down((down_proj_out_dim + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, h_NUM_TOKENS);

    kernel_down_proj_matmul<<<gridDim_down, blockDim_down>>>(
        X->d_fp16_tensor, L3_Layer->mlp_down_proj->d_fp16_tensor,
        Cache->d_feedforward_cache, down_proj_out_dim);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_gate_up_proj(
    float *d_feedforward_cache, __half *Proj_Up, __half *Proj_Gate, __half *X) {
    int token_idx = blockIdx.z;
    int fcoord_idx = blockIdx.y;
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_mem[];

    // Ensure we are within valid index bounds
    if (token_idx >= d_NUM_TOKENS || fcoord_idx >= EMBED_SIZE || embed_idx >= EMBED_SIZE) {
        return;
    }

    // Compute gate projection
    float x = __half2float(X[token_idx * EMBED_SIZE + embed_idx]);
    float gate_w = __half2float(Proj_Gate[fcoord_idx * EMBED_SIZE + embed_idx]);
    float gate_proj = x * gate_w;

    // Compute up projection
    float up_w = __half2float(Proj_Up[fcoord_idx * EMBED_SIZE + embed_idx]);
    float up_proj = x * up_w;

    // Store partial sums in shared memory for reduction
    shared_mem[threadIdx.x] = gate_proj;
    shared_mem[blockDim.x + threadIdx.x] = up_proj;
    __syncthreads();

    // Parallel reduction within the block for both gate and up
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
            shared_mem[blockDim.x + threadIdx.x] += shared_mem[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Apply SiLU to the gate projection and add to the up projection
    if (threadIdx.x == 0) {
        int idx = token_idx * EMBED_SIZE + fcoord_idx;

        // Final gate and up projections after reduction
        float gate_final = shared_mem[0];
        float up_final = shared_mem[blockDim.x];

        // Apply SiLU to gate projection
        float sigmoid = 1.0f / (1.0f + expf(-gate_final));
        float silu_gate = gate_final * sigmoid;

        // Add SiLU(gate) and up projection
        d_feedforward_cache[idx] = silu_gate + up_final;
    }
}

__global__ void kernel_down_proj_matmul(
    __half *X_out, __half *Proj_Down, float *d_feedforward_cache, int down_proj_out_dim) {
    int token_idx = blockIdx.y;
    int down_proj_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS || down_proj_idx >= down_proj_out_dim) {
        return;
    }

    // Compute the dot product between d_feedforward_cache and the down projection matrix Proj_Down
    float result = 0.0f;
    for (int i = 0; i < EMBED_SIZE; ++i) {
        float cache_val = d_feedforward_cache[token_idx * EMBED_SIZE + i];
        float down_proj_val = __half2float(Proj_Down[down_proj_idx * EMBED_SIZE + i]);
        result += cache_val * down_proj_val;
    }

    // Store the result back in X_out
    X_out[token_idx * down_proj_out_dim + down_proj_idx] = __float2half(result);
}

/* ********************************* Language Model Head ********************************* */
void compute_lm_head(Tensor *X, Tensor *LM_HEAD, CudaCache *Cache) {
    int out_dim = LM_HEAD->shape[0];
    dim3 blockDim_down(MAX_THREADS_PER_BLOCK);
    dim3 gridDim_down((out_dim + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, h_NUM_TOKENS);

    kernel_lm_head<<<gridDim_down, blockDim_down>>>(
        Cache->next_token, LM_HEAD->d_fp16_tensor,
        X->d_fp16_tensor, out_dim);
    cudaDeviceSynchronize();

    check_embedding<<<1, 1>>>(Cache->next_token, 128256);
    cudaDeviceSynchronize();

    return;
}

__global__ void kernel_lm_head(
    float *X_out, __half *LM_HEAD, __half *d_feedforward_cache, int down_proj_out_dim) {
    int token_idx = blockIdx.y;
    int down_proj_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= d_NUM_TOKENS || down_proj_idx >= down_proj_out_dim) {
        return;
    }

    float result = 0.0f;
    for (int i = 0; i < EMBED_SIZE; ++i) {
        float cache_val = d_feedforward_cache[token_idx * EMBED_SIZE + i];
        float down_proj_val = __half2float(LM_HEAD[down_proj_idx * EMBED_SIZE + i]);
        result += cache_val * down_proj_val;
    }

    // Store the result back in X_out
    X_out[token_idx * down_proj_out_dim + down_proj_idx] = __float2half(result);
}