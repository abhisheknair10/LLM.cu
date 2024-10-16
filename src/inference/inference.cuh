#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"

typedef struct {
    float *d_gnorm_cache;
    float *d_attq_cache;
    float *d_attk_cache;
    float *d_attv_cache;

    Tensor *PN_X;
    Tensor *Q;
    Tensor *K;
    Tensor *V;
} CudaCache;

/* ******************************** Inference Code ******************************** */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache);

/* *************************** Convert Tokens to Embeddings *************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

__global__ void kernel_tokens_to_embeddings(__half *fp16_tensor, __half *Embed, int *tokens);

/* ******************************* Layer Normalization ******************************* */
Tensor *_create_intermediary_prenorm_tensor_copy();

void copy_fp16_tensor(Tensor *Y, Tensor *X);

void compute_layer_norm(Tensor *RMSNorm, Tensor *X, float *d_gcache);

__global__ void kernel_compute_rms_norm(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

__global__ void kernel_compute_norm_tensor(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

/* ******************************* Attention Computation ******************************* */
Tensor *_create_intermediary_attention_tensor(Tensor *Linear);

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V,
                         Llama3Layer *L3_Layer, Tensor *X,
                         CudaCache *Cache);

void _abstract_intermediate_attensor_kernel_call(Tensor *Proj_Layer, Tensor *X, float *d_gcache);

__global__ void kernel_compute_intermediate_attention_matmul(
    __half *Linear_tensor, int *Linear_shape,
    __half *X_tensor, float *d_gcache);

void _abstract_full_attensor_kernel_call(Tensor *Attention_Tensor, Tensor *Proj_Layer, float *d_gcache);

__global__ void kernel_compute_full_attention_tensors(
    __half *O_tensor, int *Linear_shape, float *d_gcache);

/* ************************ Rotary Positional Embedding (RoPE) ************************ */
void rope_scaling(Tensor *Q, Tensor *K);

__global__ void kernel_rope_scaling(__half *tensor);

/* ************************************** Cuda Cache ************************************** */
CudaCache *init_cache(Llama3 *llama3_model);