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

    float *d_attention_score_cache;
    float *d_feedforward_cache;

    float *next_token;
} CudaCache;

/* ********************************* Inference Code ********************************* */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache);

/* ************************** Convert Tokens to Embeddings ************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

__global__ void kernel_tokens_to_embeddings(__half *fp16_tensor, __half *Embed, int *tokens);

/* ******************************* Layer Normalization ******************************* */
Tensor *_create_intermediary_prenorm_tensor_copy();

void copy_fp16_tensor(Tensor *Y, Tensor *X);

void compute_layer_norm(Tensor *RMSNorm, Tensor *X, float *d_gcache);

__global__ void kernel_compute_rms_norm(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

__global__ void kernel_compute_norm_tensor(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

void add_norm(Tensor *X, Tensor *PN_X);

__global__ void add_norm(__half *X, __half *PN_X);

/* ***************************** Attention Tensor Computation **************************** */
Tensor *_create_intermediary_attention_tensor(Tensor *Linear);

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V,
                         Llama3Layer *L3_Layer, Tensor *X,
                         CudaCache *Cache);

void compute_output(Llama3Layer *L3_Layer, Tensor *X, CudaCache *Cache);

void _abstract_intermediate_attensor_kernel_call(Tensor *Proj_Layer, Tensor *X, float *d_gcache);

__global__ void kernel_compute_intermediate_attention_matmul(
    __half *Linear_tensor, int *Linear_shape,
    __half *X_tensor, float *d_gcache);

void _abstract_full_attensor_kernel_call(Tensor *Attention_Tensor, Tensor *Proj_Layer, float *d_gcache);

__global__ void kernel_compute_full_attention_tensors(
    __half *O_tensor, int *Linear_shape, float *d_gcache);

/* ************************* Rotary Positional Embedding (RoPE) ************************* */
void rope_scaling(Tensor *Q, Tensor *K);

__global__ void kernel_rope_scaling(__half *tensor);

/* **************************** Grouped Multi-Query Attention **************************** */
void compute_attention(Tensor *X, Tensor *Q, Tensor *K, Tensor *V, CudaCache *Cache);

__global__ void kernel_compute_attention_scores_softmax(
    float *attention_scores, __half *Q, __half *K,
    int num_tokens, int nheads, int nkheads, int head_dim,
    uint8_t scaling, uint8_t automask);

__global__ void kernel_compute_attention_output(
    __half *output, float *attention_scores, __half *V,
    int num_tokens, int nheads, int nkheads, int head_dim);

/* ********************************* Feed Forward Network ********************************* */
void compute_feedforward(Tensor *X, Llama3Layer *L3_Layer, CudaCache *Cache);

__global__ void kernel_gate_up_proj(
    float *d_feedforward_cache, __half *Proj_Up, __half *Proj_Gate, __half *X);

__global__ void kernel_down_proj_matmul(
    __half *X_out, __half *Proj_Down, float *d_feedforward_cache, int down_proj_out_dim);

/* ********************************* Language Model Head ********************************* */
void compute_lm_head(Tensor *X, Tensor *LM_HEAD, CudaCache *Cache);

__global__ void kernel_lm_head(
    float *X_out, __half *LM_HEAD, __half *d_feedforward_cache, int down_proj_out_dim);

/* ************************************** Cuda Cache ************************************** */
CudaCache *init_cache(Llama3 *llama3_model);