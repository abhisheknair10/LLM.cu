#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"

/* ******************************** Inference Code ******************************** */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens);

/* *************************** Convert Tokens to Embeddings *************************** */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

__global__ void kernel_tokens_to_embeddings(__half *fp16_tensor, __half *Embed, int *tokens);

/* ******************************* Layer Normalization ******************************* */
void _create_intermediary_prenorm_tensor_copy(Tensor *Y, Tensor *X);

void copy_fp16_tensor(Tensor *Y, Tensor *X);

void compute_layer_norm(Tensor *RMSNorm, Tensor *X, float *d_gcache);

__global__ void kernel_compute_rms_norm(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

__global__ void kernel_compute_norm_tensor(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

/* ******************************* Attention Computation ******************************* */
void _create_intermediary_attention_tensor(Tensor *Attention_Tensor, Tensor *Linear);

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V,
                         Llama3Layer *L3_Layer, Tensor *X, float *d_gcache);

void _abstract_intermediate_attensor_kernel_call(Tensor *Proj_Layer, Tensor *X,
                                                 float *d_gcache, int qkv_idx);

__global__ void kernel_compute_intermediate_attention_matmul(
    __half *Linear_tensor, int *Linear_shape,
    __half *X_tensor, float *d_gcache, int qkv_idx);

void _abstract_full_attensor_kernel_call(Tensor *Attention_Tensor, Tensor *Proj_Layer,
                                         Tensor *X, float *d_gcache, int qkv_idx);

__global__ void kernel_compute_full_attention_tensors(
    __half *O_tensor, int *Linear_shape,
    float *d_gcache, int qkv_idx);