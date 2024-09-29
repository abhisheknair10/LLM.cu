#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"
#include "safetensor/safetensor.cuh"

#define WARN "\033[1;33m"
#define GREEN "\033[1;32m"
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

// CUDA kernel to check the 0th index of the fp16 tensor in the k_proj
__global__ void checker(long *mem_len) {
    // printf("The 0th index of the fp16_tensor (self_attn_k_proj): %f\n", __half2float(fp16_tensor[0]));
    printf("Mem Len: %lu\n", mem_len);
}

int main() {
    // Initialize the Llama3 model
    Llama3 *llama3_model = init_LLaMa3(MODEL_NUM_LAYERS);

    if (llama3_model == NULL) {
        printf("An error occurred while allocating memory for the Llama3 struct\n");
        exit(1);
    } else {
        printf("Model has been allocated with %d layers\n", llama3_model->n_layers);
    }

    // Load the safetensor weights into the model
    load_safetensor_weights(llama3_model, "model_weights/model-00001-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00002-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00003-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00004-of-00004.safetensors");
    CLEAR_TERMINAL();

    printf(WARN "[CPU]" RESET " Loaded model to CPU\n");
    printf(WARN "[CPU]" RESET " Moving model to CUDA and converting model parameters from BF16 to FP16\n");

    to_cuda(llama3_model);
    bf16_to_fp16(llama3_model);
    printf(GREEN "[CUDA]" RESET " Loaded to CUDA Device and formatted Parameters to FP16\n");

    // Check the 0th index of the k_proj tensor of the first layer
    checker<<<1, 1>>>(llama3_model->layers[0]->mlp_down_proj->mem_len);
    cudaDeviceSynchronize();

    // Free the model resources
    free_LLaMa3(llama3_model);

    return 0;
}
