#pragma once

#include <stdint.h>

typedef struct {
    int ndim;
    int *shape;
    long mem_len;
    uint16_t *tensor;
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

int arr_to_mem_index(Tensor *t, int n, int *index);