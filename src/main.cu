#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Llama3/Llama3.cuh"
#include "SafeTensor/SafeTensor.cuh"

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
__global__ void checker(__half *fp16_tensor) {
    printf("The 0th index of the fp16_tensor (self_attn_k_proj): %f\n", __half2float(fp16_tensor[0]));
}

int main() {
    // Initialize the Llama3 model
    Llama3 *llama3_model = init_LLaMa3(MODEL_NUM_LAYERS);

    if (llama3_model == NULL) {
        printf("An Error Occurred while allocating memory for the Llama3 Struct\n");
        exit(1);
    } else {
        printf("Model has been allocated with %d layers\n", llama3_model->n_layers);
    }

    // Load the safetensor weights into the model
    load_safetensor_weights(llama3_model, "model_weights/model-00001-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00002-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00003-of-00004.safetensors");
    load_safetensor_weights(llama3_model, "model_weights/model-00004-of-00004.safetensors");
    printf(WARN "[CPU]" RESET " Loaded Model to CPU\n");

    // Clear terminal and move the model to CUDA
    CLEAR_TERMINAL();
    printf(WARN "[CPU]" RESET " Moving Model to CUDA\n");

    to_cuda(llama3_model);
    bf16_to_fp16(llama3_model);
    printf(GREEN "[CUDA]" RESET " Formatted Parameters to FP16 and Loaded to CUDA Device\n");

    // Check the 0th index of the k_proj tensor of the first layer
    printf(GREEN "[CUDA]" RESET " Launching CUDA checker kernel for 0th index of k_proj in first layer\n");
    checker<<<1, 1>>>(llama3_model->layers[0]->self_attn_k_proj->d_fp16_tensor);
    cudaDeviceSynchronize();

    // Free the model resources
    free_LLaMa3(llama3_model);

    return 0;
}
