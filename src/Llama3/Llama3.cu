#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Llama3.cuh"

Llama3 *init_LLaMa3(int n_layers) {
    // main llama 3 struct with n layers
    Llama3 *llama3 = (Llama3 *)malloc(sizeof(Llama3));
    if (llama3 == NULL) {
        printf("An Error Occurred while allocating memory for the Llama3 Struct\n");
        exit(1);
    }

    // number of decoder layers
    llama3->n_layers = n_layers;

    // allocate embed and head layers
    llama3->embed_tokens = (Tensor *)malloc(sizeof(Tensor));
    llama3->norm = (Tensor *)malloc(sizeof(Tensor));
    llama3->lm_head = (Tensor *)malloc(sizeof(Tensor));
    if (llama3->embed_tokens == NULL || llama3->norm == NULL || llama3->lm_head == NULL) {
        printf("An Error Occurred while allocating memory for a Llama3 Layer\n");
        exit(1);
    }

    // allocate layer pointers
    llama3->layers = (Llama3Layer **)malloc(sizeof(Llama3Layer *) * n_layers);
    if (llama3->layers == NULL) {
        printf("An Error Occurred while allocating memory for a Llama3 Layer\n");
        exit(1);
    }

    for (int i = 0; i < n_layers; i++) {
        Llama3Layer *layer = (Llama3Layer *)malloc(sizeof(Llama3Layer));
        if (layer == NULL) {
            printf("An Error Occurred while allocating memory for a Llama3 Layer\n");
            exit(1);
        }

        // Initialize Tensors for the layer if needed
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

        if (layer->input_layernorm == NULL || layer->mlp_down_proj == NULL ||
            layer->mlp_gate_proj == NULL || layer->mlp_up_proj == NULL ||
            layer->post_attention_layernorm == NULL || layer->self_attn_k_proj == NULL ||
            layer->self_attn_o_proj == NULL || layer->self_attn_q_proj == NULL ||
            layer->self_attn_v_proj == NULL) {
            printf("An Error Occurred while allocating memory for a Llama3 Layer\n");
            exit(1);
        }

        llama3->layers[i] = layer;
    }

    return llama3;
}

void free_LLaMa3(Llama3 *llama3) {
    free(llama3->embed_tokens);
    free(llama3->lm_head);

    // Free each layer
    for (int i = 0; i < llama3->n_layers; i++) {
        Llama3Layer *layer = llama3->layers[i];

        // Free each Tensor within the layer
        free(layer->input_layernorm);
        free(layer->mlp_down_proj);
        free(layer->mlp_gate_proj);
        free(layer->mlp_up_proj);
        free(layer->post_attention_layernorm);
        free(layer->self_attn_k_proj);
        free(layer->self_attn_o_proj);
        free(layer->self_attn_q_proj);
        free(layer->self_attn_v_proj);

        free(layer);
    }

    free(llama3->layers);
    free(llama3);
}

typedef struct {
    Tensor *embed_tokens;
    Tensor *lm_head;
    Tensor *norm;
    Llama3Layer **layers;
    int n_layers;
} Llama3;

/*
To move the following:
    - All members in the Tensor struct, will be used for matrix multiplication
    - Since pointers for higher level functions can be passed very quickly but also seldomly,
        we keep the rest on the CPU
*/
void to_cuda(Llama3 *llama3) {
    for (int i = 0; i < llama3->n_layers; i++) {
        helper_move_tensor_to_cuda(llama3->layers[i]->input_layernorm);
        helper_move_tensor_to_cuda(llama3->layers[i]->mlp_down_proj);
        helper_move_tensor_to_cuda(llama3->layers[i]->mlp_gate_proj);
        helper_move_tensor_to_cuda(llama3->layers[i]->mlp_up_proj);
        helper_move_tensor_to_cuda(llama3->layers[i]->post_attention_layernorm);
        helper_move_tensor_to_cuda(llama3->layers[i]->self_attn_k_proj);
        helper_move_tensor_to_cuda(llama3->layers[i]->self_attn_o_proj);
        helper_move_tensor_to_cuda(llama3->layers[i]->self_attn_q_proj);
        helper_move_tensor_to_cuda(llama3->layers[i]->self_attn_v_proj);
    }

    return;
}

void helper_move_tensor_to_cuda(Tensor *tensor) {
    int *d_ndim;
    long *d_mem_len;
    int **d_shape;
    uint16_t **d_bf16_tensor;
    __half **d_fp16_tensor;

    cudaMalloc((void **)&d_ndim, sizeof(int));
    cudaMalloc((void **)&d_mem_len, sizeof(long));

    cudaMemcpy(d_ndim, tensor->ndim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_len, tensor->mem_len, sizeof(long), cudaMemcpyHostToDevice);
}

int arr_to_mem_index(Tensor *t, int n, int *idx) {
    int mem_idx = 0;
    int stride = 1;

    for (int i = n - 1; i >= 0; i--) {
        mem_idx += (idx[i] * stride);
        stride *= t->shape[i];
    }

    return mem_idx;
}