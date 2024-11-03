#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"

typedef struct {
    Tensor *PN_X;

    Tensor *Q;
    Tensor *K;
    Tensor *V;

    float *d_attention_score_cache;
    float *d_feedforward_cache;

    __half *next_token;
} CudaCache;

/* ********************************* Inference Code ********************************* */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens, CudaCache *Cache);

/* ************************** Convert Tokens to Embeddings ************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

__global__ void kernel_tokens_to_embeddings(__half *X, int *tokens, __half *Embed);

/* ******************************* Layer Normalization ******************************* */
Tensor *_create_intermediary_prenorm_tensor_copy();

void _deviceMemcpy_fp16_tensor(Tensor *Y, Tensor *X);

void compute_layer_norm(Tensor *RMSNorm, Tensor *X);

__global__ void kernel_compute_rms_norm(__half *X, __half *RMSNorm);

void add_norm(Tensor *X, Tensor *PN_X);

__global__ void add_norm(__half *X, __half *PN_X);

/* ***************************** General Matrix Multiplication **************************** */
__global__ void kernel_standard_tiled_gemm(
    __half *O, __half *X, __half *Transform, int m, int n, int k, int TILE_SIZE);

/* ***************************** Attention Tensor Computation **************************** */
Tensor *_create_intermediary_attention_tensor(Tensor *Linear);

void compute_qkv_tensors(
    Tensor *Q, Tensor *K, Tensor *V,
    Llama3Layer *L3_Layer, Tensor *X);

void compute_output(Llama3Layer *L3_Layer, Tensor *X);

/* ************************* Rotary Positional Embedding (RoPE) ************************* */
void rope_scaling(Tensor *Q, Tensor *K);

__global__ void kernel_rope_scaling(__half *tensor, int transformed_embed_size);

/* **************************** Grouped Multi-Query Attention **************************** */
void compute_attention(Tensor *X, Tensor *Q, Tensor *K, Tensor *V, CudaCache *Cache);

__global__ void kernel_compute_masked_gmq_attention_scores_tiled_matmul(
    float *attention_scores, __half *Q, __half *K,
    int m, int n, int k, int TILE_SIZE, int nheads);

__global__ void kernel_masking_softmax(__half *attention_scores, int masking, int softmax);

__global__ void kernel_compute_resolved_value_from_attention_score_tiled_matmul(
    __half *output, float *attention_scores, __half *V,
    int m, int d_head, int nheads, int TILE_SIZE);

/* ********************************* Feed Forward Network ********************************* */
void compute_feedforward(Tensor *X, Llama3Layer *L3_Layer, CudaCache *Cache);

__global__ void kernel_gate_up_proj(
    float *d_feedforward_cache, __half *Proj_Up, __half *Proj_Gate, __half *X);

__global__ void kernel_down_proj_matmul(
    __half *X_out, __half *Proj_Down, float *d_feedforward_cache, int down_proj_out_dim);

/* ********************************* Language Model Head ********************************* */
void compute_lm_head(Tensor *X, Tensor *LM_Head, CudaCache *Cache);

/* ************************************** Cuda Cache ************************************** */
CudaCache *init_cache(Llama3 *llama3_model);