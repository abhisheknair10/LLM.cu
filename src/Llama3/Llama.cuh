#pragma once

#include <stdint.h>

typedef struct {
    int d_ndim;
    int *d_shape;
    long d_mem_len;
    half *d_tensor;
} d_Tensor;

typedef struct {
    int layernum;
    d_Tensor *d_input_layernorm;
    d_Tensor *d_mlp_down_proj;
    d_Tensor *d_mlp_gate_proj;
    d_Tensor *d_mlp_up_proj;
    d_Tensor *d_post_attention_layernorm;
    d_Tensor *d_self_attn_k_proj;
    d_Tensor *d_self_attn_o_proj;
    d_Tensor *d_self_attn_q_proj;
    d_Tensor *d_self_attn_v_proj;
} d_Llama3Layer;

typedef struct {
    d_Tensor *d_embed_tokens;
    d_Tensor *d_lm_head;
    d_Tensor *d_norm;
    d_Llama3Layer **d_layers;
    int n_layers;
} d_Llama3;