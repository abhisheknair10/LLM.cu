#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"

void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens);

void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

__global__ void kernel_tokens_to_embeddings(__half *fp16_tensor, __half *Embed, int *tokens);

void _create_intermediary_attention_tensor(Tensor *Attention_Tensor, Tensor *Linear);

void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V, Llama3Layer *L3_Layer, Tensor *X);

__global__ void kernel_compute_attention_tensors(Tensor *O_tensor, Tensor *Linear, Tensor *X);