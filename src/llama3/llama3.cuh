#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

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
#ifdef __CUDACC__
    __half *d_fp16_tensor;
#endif
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
 * This function deallocates all the memory used by the Llama3 structure,
 * including all layers and tensors.
 *
 * @param llama3 Pointer to the Llama3 structure to be freed.
 */
#ifdef __CUDACC__
void free_LLaMa3(Llama3 *llama3);
#endif

/**
 * @brief Frees memory allocated for a Tensor structure.
 *
 * This function frees both the CPU and CUDA memory associated with the Tensor structure.
 *
 * @param tensor Pointer to the Tensor structure to be freed.
 */
#ifdef __CUDACC__
void _free_tensor(Tensor *tensor);
#endif

/**
 * @brief Transfers the Llama3 model's data from CPU to CUDA memory.
 *
 * This function transfers all tensors in the model to CUDA memory.
 *
 * @param llama3 Pointer to the Llama3 structure to transfer to CUDA.
 */
void to_cuda(Llama3 *llama3);

/**
 * @brief Transfers a Tensor's data from CPU to CUDA memory.
 *
 * This function transfers the specified tensor's data (dimensions, shape, and values) to CUDA memory.
 *
 * @param tensor Pointer to the Tensor structure to transfer to CUDA.
 */
#ifdef __CUDACC__
void _move_tensor_to_cuda(Tensor *tensor);
#endif

/**
 * @brief Converts the bf16 data in the Llama3 model to fp16.
 *
 * This function allocates memory for fp16 data and launches the necessary kernels
 * to convert bf16 data to fp16 for each tensor in the Llama3 model.
 *
 * @param llama3 Pointer to the Llama3 structure.
 */

void bf16_to_fp16(Llama3 *llama3);

/**
 * @brief Allocates fp16 memory for a specific tensor in CUDA memory.
 *
 * This function allocates memory on the GPU for storing fp16 data (__half) for a specific tensor.
 *
 * @param tensor Pointer to the Tensor structure.
 */
#ifdef __CUDACC__
void _cudaMalloc_fp16(Tensor *tensor);
#endif

/**
 * @brief Wrapper function to launch the bf16 to fp16 conversion kernel.
 *
 * This function launches the CUDA kernel to convert the bf16 tensor to fp16 on the device.
 * It ensures that all elements in the tensor are processed by calculating appropriate thread and block dimensions.
 *
 * @param tensor Pointer to the Tensor structure that contains the bf16 data.
 */
#ifdef __CUDACC__
void _kernel_wrapper_bf16_to_fp16(Tensor *tensor);
#endif

/**
 * @brief CUDA kernel that converts bfloat16 (bf16) tensor data to float16 (fp16).
 *
 * This kernel processes each element of the input tensor, converting it from bf16 (bfloat16) format
 * to fp16 (float16) format. Each thread handles the conversion of one element, up to the total number
 * of elements specified by the `mem_len` parameter.
 *
 * @param bf16_tensor Pointer to the input tensor data in bf16 format (16-bit unsigned integers, uint16_t).
 *                    This must reside in device (CUDA) memory and be properly allocated.
 * @param fp16_tensor Pointer to the output tensor data in fp16 format (__half). The converted data will
 *                    be stored in this array. This must also reside in device (CUDA) memory and be allocated
 *                    to match the size of `bf16_tensor`.
 * @param mem_len Pointer to the number of elements in the input/output tensors. This value indicates how many
 *                elements are to be processed by the kernel. The kernel will process up to `*mem_len` elements.
 *                It must reside in device memory.
 */
#ifdef __CUDACC__
__global__ void _kernel_bf16_to_fp16(uint16_t *bf16_tensor, __half *fp16_tensor, long *mem_len);
#endif

/**
 * @brief Performs an operation on each tensor in the Llama3 model.
 *
 * This function applies a user-defined function to each tensor within the layers of the Llama3 model.
 *
 * @param llama3 Pointer to the Llama3 structure.
 * @param _func Function to apply to each tensor in the model.
 */
void _m_component_tensor_operation(Llama3 *llama3, void (*_func)(Tensor *));

/**
 * @brief Converts an array of indices to a memory index in a tensor.
 *
 * This function computes the memory index corresponding to a set of indices (idx)
 * in a tensor based on its shape. It converts multi-dimensional array indices to a flat memory index.
 *
 * @param t Pointer to the Tensor structure.
 * @param n Number of dimensions in the tensor.
 * @param idx Array of indices to convert to a memory index.
 * @return The computed memory index.
 */
int arr_to_mem_index(Tensor *t, int n, int *idx);
