#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Llama3.cuh"

Llama3 *init_LLaMa3(int n_layers) {
    // Allocate memory for the Llama3 model
    Llama3 *llama3 = (Llama3 *)malloc(sizeof(Llama3));
    if (llama3 == NULL) {
        printf("Error: Memory allocation failed for Llama3 structure.\n");
        exit(1);
    }

    llama3->n_layers = n_layers;

    // Allocate embed, norm, and lm_head tensors
    llama3->embed_tokens = (Tensor *)malloc(sizeof(Tensor));
    llama3->norm = (Tensor *)malloc(sizeof(Tensor));
    llama3->lm_head = (Tensor *)malloc(sizeof(Tensor));
    if (llama3->embed_tokens == NULL || llama3->norm == NULL || llama3->lm_head == NULL) {
        printf("Error: Memory allocation failed for embed, norm, or lm_head tensors.\n");
        exit(1);
    }

    // Allocate and initialize each layer
    llama3->layers = (Llama3Layer **)malloc(sizeof(Llama3Layer *) * n_layers);
    if (llama3->layers == NULL) {
        printf("Error: Memory allocation failed for layers array.\n");
        exit(1);
    }

    for (int i = 0; i < n_layers; i++) {
        Llama3Layer *layer = (Llama3Layer *)malloc(sizeof(Llama3Layer));
        if (layer == NULL) {
            printf("Error: Memory allocation failed for layer %d.\n", i);
            exit(1);
        }

        // Initialize each tensor in the layer
        layer->input_layernorm = (Tensor *)malloc(sizeof(Tensor));
        layer->mlp_down_proj = (Tensor *)malloc(sizeof(Tensor));
        layer->mlp_gate_proj = (Tensor *)malloc(sizeof(Tensor));
        layer->mlp_up_proj = (Tensor *)malloc(sizeof(Tensor));
        layer->post_attention_layernorm = (Tensor *)malloc(sizeof(Tensor));
        layer->self_attn_k_proj = (Tensor *)malloc(sizeof(Tensor));
        layer->self_attn_o_proj = (Tensor *)malloc(sizeof(Tensor));
        layer->self_attn_q_proj = (Tensor *)malloc(sizeof(Tensor));
        layer->self_attn_v_proj = (Tensor *)malloc(sizeof(Tensor));

        layer->layernum = i;

        // Check for allocation failures for each tensor
        if (layer->input_layernorm == NULL || layer->mlp_down_proj == NULL ||
            layer->mlp_gate_proj == NULL || layer->mlp_up_proj == NULL ||
            layer->post_attention_layernorm == NULL || layer->self_attn_k_proj == NULL ||
            layer->self_attn_o_proj == NULL || layer->self_attn_q_proj == NULL ||
            layer->self_attn_v_proj == NULL) {
            printf("Error: Memory allocation failed for tensors in layer %d.\n", i);
            exit(1);
        }

        llama3->layers[i] = layer;
    }

    return llama3;
}

void free_LLaMa3(Llama3 *llama3) {
    // Free non-layer tensors
    free(llama3->embed_tokens);
    free(llama3->norm);
    free(llama3->lm_head);

    // Free each tensor inside the layers
    _m_component_tensor_operation(llama3, _free_tensor);

    // Free each layer structure
    for (int i = 0; i < llama3->n_layers; i++) {
        free(llama3->layers[i]);
    }

    // Free the layers array and the Llama3 structure
    free(llama3->layers);
    free(llama3);
}

void _free_tensor(Tensor *tensor) {
    if (tensor == NULL) return;

    // CUDA memory
    if (tensor->d_ndim) cudaFree(tensor->d_ndim);
    if (tensor->d_mem_len) cudaFree(tensor->d_mem_len);
    if (tensor->d_shape) cudaFree(tensor->d_shape);
    if (tensor->d_bf16_tensor) cudaFree(tensor->d_bf16_tensor);
    if (tensor->d_fp16_tensor) cudaFree(tensor->d_fp16_tensor);

    // CPU memory
    if (tensor->ndim) free(tensor->ndim);
    if (tensor->mem_len) free(tensor->mem_len);
    if (tensor->shape) free(tensor->shape);
    if (tensor->bf16_tensor) free(tensor->bf16_tensor);

    free(tensor);
}

void to_cuda(Llama3 *llama3) {
    _m_component_tensor_operation(llama3, _move_tensor_to_cuda);
}

void _move_tensor_to_cuda(Tensor *tensor) {
    int *d_ndim;
    long *d_mem_len;
    int *d_shape;
    uint16_t *d_bf16_tensor;

    // Allocate GPU memory
    cudaMalloc((void **)&d_ndim, sizeof(int));
    cudaMalloc((void **)&d_mem_len, sizeof(long));
    cudaMalloc((void **)&d_shape, sizeof(int) * (*(tensor->ndim)));
    cudaMalloc((void **)&d_bf16_tensor, sizeof(uint16_t) * (*(tensor->mem_len)));

    // Copy data from CPU to GPU
    cudaMemcpy(d_ndim, tensor->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, tensor->mem_len, sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, tensor->shape, sizeof(int) * (*(tensor->ndim)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bf16_tensor, tensor->bf16_tensor, sizeof(uint16_t) * (*(tensor->mem_len)), cudaMemcpyHostToDevice);

    // Free the CPU memory after transfer
    free(tensor->bf16_tensor);

    // Update tensor pointers to CUDA memory
    tensor->d_ndim = d_ndim;
    tensor->d_mem_len = d_mem_len;
    tensor->d_shape = d_shape;
    tensor->d_bf16_tensor = d_bf16_tensor;
}

void bf16_to_fp16(Llama3 *llama3) {
    _m_component_tensor_operation(llama3, _cudaMalloc_fp16);
    _m_component_tensor_operation(llama3, _kernel_wrapper_bf16_to_fp16);
}

void _cudaMalloc_fp16(Tensor *tensor) {
    __half *d_fp16_tensor;

    // Allocate fp16 tensor memory on the GPU
    cudaMalloc((void **)&d_fp16_tensor, sizeof(__half) * (*(tensor->mem_len)));

    tensor->d_fp16_tensor = d_fp16_tensor;
}

void _kernel_wrapper_bf16_to_fp16(Tensor *tensor) {
    if (tensor->d_bf16_tensor == NULL) {
        printf("Error: Expected BF16 Tensor on Device to be allocated\n");
        exit(1);
    }

    int threads_per_block = 1024;
    int num_blocks = ((*(tensor->mem_len)) + threads_per_block - 1) / threads_per_block;

    _kernel_bf16_to_fp16<<<num_blocks, threads_per_block>>>(
        tensor->d_bf16_tensor, tensor->d_fp16_tensor, tensor->d_mem_len);

    cudaDeviceSynchronize();
    cudaFree(tensor->d_bf16_tensor);
}

__global__ void _kernel_bf16_to_fp16(uint16_t *bf16_tensor, __half *fp16_tensor, long *mem_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < *mem_len) {
        // BF16 to FP32
        uint32_t bf16 = (uint32_t)bf16_tensor[idx];
        uint32_t fp32 = bf16 << 16;
        float fp32_value = __uint2float_rn(fp32);

        // FP32 to FP16
        fp16_tensor[idx] = __float2half_rn(fp32_value);
    }
}

// Applies a user-defined function to each tensor in the Llama3 model.
void _m_component_tensor_operation(Llama3 *llama3, void (*_func)(Tensor *)) {
    for (int i = 0; i < llama3->n_layers; i++) {
        _func(llama3->layers[i]->input_layernorm);
        _func(llama3->layers[i]->mlp_down_proj);
        _func(llama3->layers[i]->mlp_gate_proj);
        _func(llama3->layers[i]->mlp_up_proj);
        _func(llama3->layers[i]->post_attention_layernorm);
        _func(llama3->layers[i]->self_attn_k_proj);
        _func(llama3->layers[i]->self_attn_o_proj);
        _func(llama3->layers[i]->self_attn_q_proj);
        _func(llama3->layers[i]->self_attn_v_proj);
    }
}

int arr_to_mem_index(Tensor *t, int n, int *idx) {
    int mem_idx = 0;
    int stride = 1;

    // Compute the memory index using the tensor shape
    for (int i = n - 1; i >= 0; i--) {
        mem_idx += (idx[i] * stride);
        stride *= t->shape[i];
    }

    return mem_idx;
}
