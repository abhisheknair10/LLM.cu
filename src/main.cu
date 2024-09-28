#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Llama3/Llama3.cuh"
#include "SafeTensor/SafeTensor.cuh"

#define WARN "\033[1;33m"
#define GREY "\033[2m"
#define RESET "\033[0m"

#ifdef _WIN32
#define CLEAR_TERMINAL() system("cls")
#elif __linux__ || __APPLE__
#define CLEAR_TERMINAL() system("clear")
#else
#define CLEAR_TERMINAL() printf("\n")
#endif

const int MODEL_NUM_LAYERS = 32;

__global__ void print_cuda_mem(uint16_t *b16_tensor) {
    // Extract the 1st index of the bfloat16 tensor
    uint16_t bf16_value = b16_tensor[1];

    // Convert bfloat16 to float32
    // Shift the bfloat16 (16 bits) into the upper 16 bits of a 32-bit float representation
    uint32_t fp32_value_bits = (uint32_t)bf16_value << 16;

    // Reinterpret the bits as a float
    float fp32_value = __int_as_float(fp32_value_bits);

    // Print the converted float value
    printf("Converted fp32 value: %f\n", fp32_value);
}


int main() {
    Llama3 *llama3_model = init_LLaMa3(MODEL_NUM_LAYERS);

    if (llama3_model == NULL) {
        printf("An Error Occurred while allocating memory for the Llama3 Struct\n");
        exit(1);
    } else {
        printf("Model has been allocated with %d layers\n", llama3_model->n_layers);
    }

    load_safetensor_weights(llama3_model, "model_weights/model-00001-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00002-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00003-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00004-of-00004.safetensors");

    // CLEAR_TERMINAL();

    printf(WARN "[CPU]" RESET " Loaded Model Weights\n");

    to_cuda(llama3_model);

    print_cuda_mem<<<1, 1>>>(llama3_model->layers[0]->self_attn_k_proj->bf16_tensor);
    cudaDeviceSynchronize();

    free_LLaMa3(llama3_model);

    return 0;
}