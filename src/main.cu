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

    CLEAR_TERMINAL();

    printf(WARN "[CPU]" RESET " Loaded Model Weights\n");

    printf(WARN "[CPU]" RESET " Moving Model to CUDA\n");
    to_cuda(llama3_model);

    printf(GREEN "[CUDA]" RESET " Loaded Model Weights\n");

    free_LLaMa3(llama3_model);

    return 0;
}