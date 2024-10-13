#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"

/* ******************************** Inference Code ******************************** */
/**
 * @brief Perform inference on the LLaMA3 model.
 *
 * This function initiates the inference process, taking the model object, input tokens,
 * and output tensors to compute the next set of tokens in the sequence.
 *
 * @param llama3_model Pointer to the Llama3 model structure.
 * @param X Pointer to the input tensor (embedding of input tokens).
 * @param d_tokens Pointer to device-side tokens to be processed.
 * @param h_tokens Pointer to host-side tokens for processing.
 */
void inference(Llama3 *llama3_model, Tensor *X, int *d_tokens, int *h_tokens);

/* *************************** Convert Tokens to Embeddings *************************** */
/**
 * @brief Convert input tokens into embeddings.
 *
 * This function converts a set of input tokens into corresponding embeddings using
 * the token embedding matrix of the Llama3 model.
 *
 * @param X Pointer to the tensor where embeddings will be stored.
 * @param llama3_model Pointer to the Llama3 model structure.
 * @param d_tokens Pointer to device-side tokens to be converted.
 */
void tokens_to_embeddings(Tensor *X, Llama3 *llama3_model, int *d_tokens);

/**
 * @brief CUDA kernel to convert tokens into embeddings.
 *
 * This kernel function performs the conversion of tokens to embeddings on the device,
 * using fp16 precision.
 *
 * @param fp16_tensor Pointer to the output tensor in fp16 format.
 * @param Embed Pointer to the embedding matrix.
 * @param tokens Pointer to the tokens being converted to embeddings.
 */
__global__ void kernel_tokens_to_embeddings(__half *fp16_tensor, __half *Embed, int *tokens);

/* ******************************* Layer Normalization ******************************* */
/**
 * @brief Create an intermediary copy of the pre-normalization tensor.
 *
 * This function creates a copy of the input tensor to be used in layer normalization,
 * ensuring that intermediate results are stored safely.
 *
 * @param Y Pointer to the output tensor (intermediary).
 * @param X Pointer to the input tensor (before normalization).
 */
void _create_intermediary_prenorm_tensor_copy(Tensor *Y, Tensor *X);

/**
 * @brief Copy an fp16 tensor from one location to another.
 *
 * This function is used to copy a half-precision floating-point tensor (fp16) from one
 * tensor structure to another.
 *
 * @param Y Pointer to the destination tensor.
 * @param X Pointer to the source tensor.
 */
void copy_fp16_tensor(Tensor *Y, Tensor *X);

/**
 * @brief Compute LayerNorm using RMS normalization.
 *
 * This function calculates the LayerNorm of an input tensor using Root Mean Square (RMS)
 * normalization. The norm values are cached in a float cache on the device.
 *
 * @param RMSNorm Pointer to the output tensor (RMS normalized).
 * @param X Pointer to the input tensor (to be normalized).
 * @param d_gcache Pointer to the device-side cache for normalization values.
 */
void compute_layer_norm(Tensor *RMSNorm, Tensor *X, float *d_gcache);

/**
 * @brief CUDA kernel to compute RMS normalization.
 *
 * This kernel calculates the RMS normalization for an input tensor.
 *
 * @param X_tensor Pointer to the input tensor (fp16 format).
 * @param RMSNorm_tensor Pointer to the output tensor (RMS normalized, fp16 format).
 * @param d_gcache Pointer to the device-side cache for normalization values.
 */
__global__ void kernel_compute_rms_norm(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

/**
 * @brief CUDA kernel to compute the norm tensor for normalization.
 *
 * This kernel function calculates the norm of an input tensor, used for the final normalization
 * process in LayerNorm.
 *
 * @param X_tensor Pointer to the input tensor (fp16 format).
 * @param RMSNorm_tensor Pointer to the output tensor (fp16 format).
 * @param d_gcache Pointer to the device-side cache for normalization values.
 */
__global__ void kernel_compute_norm_tensor(__half *X_tensor, __half *RMSNorm_tensor, float *d_gcache);

/* ******************************* Attention Computation ******************************* */
/**
 * @brief Create an intermediary attention tensor for attention computations.
 *
 * This function creates a temporary tensor that holds the intermediary results for
 * attention computations.
 *
 * @param Attention_Tensor Pointer to the tensor storing intermediary attention values.
 * @param Linear Pointer to the linear projection tensor used in the attention mechanism.
 */
void _create_intermediary_attention_tensor(Tensor *Attention_Tensor, Tensor *Linear);

/**
 * @brief Compute the Q, K, V tensors for the attention mechanism.
 *
 * This function computes the query (Q), key (K), and value (V) tensors from the input
 * tensor using the attention layers of the Llama3 model.
 *
 * @param Q Pointer to the tensor storing the query vectors.
 * @param K Pointer to the tensor storing the key vectors.
 * @param V Pointer to the tensor storing the value vectors.
 * @param L3_Layer Pointer to the attention layer of the Llama3 model.
 * @param X Pointer to the input tensor to compute Q, K, V.
 */
void compute_qkv_tensors(Tensor *Q, Tensor *K, Tensor *V, Llama3Layer *L3_Layer, Tensor *X);

/**
 * @brief CUDA kernel to compute attention tensors.
 *
 * This kernel performs the computations required for the attention mechanism, transforming
 * the input tensor using the attention layers to obtain output tensors.
 *
 * @param O_tensor Pointer to the output tensor for attention results.
 * @param O_ndim Pointer to the number of dimensions of the output tensor.
 * @param O_shape Pointer to the shape array of the output tensor.
 * @param Linear_tensor Pointer to the linear projection tensor for attention.
 * @param Linear_ndim Pointer to the number of dimensions of the linear tensor.
 * @param Linear_shape Pointer to the shape array of the linear tensor.
 * @param X_tensor Pointer to the input tensor for attention.
 * @param X_ndim Pointer to the number of dimensions of the input tensor.
 * @param X_shape Pointer to the shape array of the input tensor.
 */
__global__ void kernel_compute_attention_tensors(
    __half *O_tensor, int *O_ndim, int *O_shape,
    __half *Linear_tensor, int *Linear_ndim, int *Linear_shape,
    __half *X_tensor, int *X_ndim, int *X_shape);
