#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

typedef struct {
    int *ndim;
    long *mem_len;
    int *shape;
    uint16_t *bf16_tensor;
    __half *fp16_tensor;
} Tensor;

typedef struct {
    int layernum;
    Tensor *input_layernorm;
    Tensor *mlp_down_proj;
    Tensor *mlp_gate_proj;
    Tensor *mlp_up_proj;
    Tensor *post_attention_layernorm;
    Tensor *self_attn_k_proj;
    Tensor *self_attn_o_proj;
    Tensor *self_attn_q_proj;
    Tensor *self_attn_v_proj;
} Llama3Layer;

typedef struct {
    Tensor *embed_tokens;
    Tensor *lm_head;
    Tensor *norm;
    Llama3Layer **layers;
    int n_layers;
} Llama3;

Llama3 *init_LLaMa3(int n_layers);

void free_LLaMa3(Llama3 *llama3);

void to_cuda(Llama3 *llama3);

void helper_move_tensor_to_cuda(Tensor *tensor);

int arr_to_mem_index(Tensor *t, int n, int *index);