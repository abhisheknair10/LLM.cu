#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

/**
 * @brief Tensor structure representing multidimensional data.
 */
typedef struct {
    // Host members
    int *ndim;
    long *mem_len;
    int *shape;
    uint16_t *bf16_tensor;

    // CUDA device members
    int *d_ndim;
    long *d_mem_len;
    int *d_shape;
    uint16_t *d_bf16_tensor;
    __half *d_fp16_tensor;
} Tensor;

/**
 * @brief Llama3Layer structure representing a layer in the Llama3 model.
 */
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

/**
 * @brief Llama3 structure representing the full Llama3 model.
 */
typedef struct {
    Tensor *embed_tokens;
    Tensor *lm_head;
    Tensor *norm;
    Llama3Layer **layers;
    int n_layers;
} Llama3;

/**
 * @brief Initializes the Llama3 model with the specified number of layers.
 *
 * @param n_layers Number of layers in the model.
 * @return Pointer to the initialized Llama3 structure.
 */
Llama3 *init_LLaMa3(int n_layers);

/**
 * @brief Frees the memory allocated for the Llama3 model and its components.
 *
 * @param llama3 Pointer to the Llama3 structure to be freed.
 */
void free_LLaMa3(Llama3 *llama3);

/**
 * @brief Transfers the Llama3 model data to CUDA memory.
 *
 * This function transfers each tensor in the model to CUDA memory.
 *
 * @param llama3 Pointer to the Llama3 structure to be transferred to CUDA memory.
 */
void to_cuda(Llama3 *llama3);

/**
 * @brief Moves a tensor's data to CUDA memory.
 *
 * This helper function transfers the tensor data from CPU memory to GPU memory.
 *
 * @param tensor Pointer to the Tensor structure to be moved to CUDA memory.
 */
void _move_tensor_to_cuda(Tensor *tensor);

/**
 * @brief Allocates fp16 data for the Llama3 model in CUDA memory.
 *
 * This function allocates fp16 (__half) data for each tensor in the model.
 *
 * @param llama3 Pointer to the Llama3 structure for which fp16 data will be allocated.
 */
void bf16_to_fp16(Llama3 *llama3);

/**
 * @brief Allocates fp16 data for a specific tensor in CUDA memory.
 *
 * @param tensor Pointer to the Tensor structure for which fp16 data will be allocated.
 */
void _cudaMalloc_fp16(Tensor *tensor);

/**
 * @brief A wrapper function to invoke the CUDA kernel that converts bfloat16 (bf16) tensor data to float16 (fp16).
 *
 * This function wraps the CUDA kernel that performs the conversion of tensor data from bf16 format
 * (stored in the Tensor struct) to fp16 format, utilizing the GPU for parallel computation.
 *
 * @param tensor Pointer to the Tensor structure containing the bf16 data that needs to be converted to fp16.
 *               The function assumes that the tensor's `bf16_tensor` field contains the data to be converted,
 *               and the `fp16_tensor` field will be allocated and populated with the converted fp16 data.
 */
void _kernel_wrapper_bf16_to_fp16(Tensor *tensor);

/**
 * @brief CUDA kernel that converts an input bfloat16 (bf16) tensor to float16 (fp16) format.
 *
 * This kernel is responsible for converting each element of the input tensor from bf16 (bfloat16) format
 * to fp16 (float16) format. It processes the input tensor up to the size specified by the `mem_len` parameter.
 * Each element of the input bf16 tensor is converted and stored in the corresponding position of the
 * output fp16 tensor.
 *
 * @param bf16_tensor The input tensor data in bf16 (bfloat16) format. The data is represented as 16-bit
 *                    unsigned integers (uint16_t). This must be allocated in the device memory before
 *                    the kernel is launched.
 * @param fp16_tensor The output tensor data in fp16 (float16) format. The data is represented as half-precision
 *                    floating-point values (__half). This must be allocated in the device memory before
 *                    the kernel is launched.
 * @param mem_len Pointer to a long integer specifying the number of elements in the bf16 and fp16 tensors.
 *                This value determines how many elements the kernel will process.
 */
__global__ void _kernel_bf16_to_fp16(uint16_t bf16_tensor, __half fp16_tensor, long *mem_len);

/**
 * @brief Performs a component-wise operation on each tensor in the Llama3 model.
 *
 * @param llama3 Pointer to the Llama3 structure.
 * @param _func Pointer to the function to be applied to each tensor in the model.
 */
void _m_component_tensor_operation(Llama3 *llama3, void (*_func)(Tensor));

/**
 * @brief Converts an array of indices to a memory index in a tensor.
 *
 * This function computes the memory index corresponding to a set of indices (idx)
 * in a tensor with shape stored in the Tensor struct. It converts multi-dimensional
 * array indices to a flat memory index.
 *
 * @param t Pointer to the Tensor structure.
 * @param n The number of dimensions in the tensor.
 * @param index Array of indices for which to calculate the memory index.
 * @return The calculated memory index.
 */
int arr_to_mem_index(Tensor *t, int n, int *index);