#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/inference.cuh"
#include "llama3/llama3.cuh"
#include "safetensor/safetensor.cuh"
#include "tokenizer/tokenizer.cuh"

#define WARN "\033[1;33m"
#define GREEN "\033[1;32m"
#define GREY "\033[2m"
#define RESET "\033[0m"

#define CLEAR_TERMINAL() system("clear")

const int MODEL_NUM_LAYERS = 32;

int main() {
    // Initialize the Llama3 model
    Llama3 *llama3_model = init_llama3(MODEL_NUM_LAYERS);

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

    printf(WARN "[CPU]" RESET " Loaded model to CPU\n");
    printf(WARN "[CPU]" RESET " Moving model to CUDA\n");
    printf(WARN "[CPU]" RESET " Converting from BF16 to FP16\n");

    to_cuda(llama3_model);
    printf(GREEN "[CUDA]" RESET " Loaded to CUDA Device\n");
    printf(GREEN "[CUDA]" RESET " Formatted Parameters to FP16\n");

    // Load the tokenizer (this function should load the trie from the tokenizer's JSON)
    Llama3Tokenizer *llama3_tokenizer = load_tokenizer();
    if (llama3_tokenizer == NULL) {
        printf("Error: Failed to load the tokenizer\n");
        return 1;
    }

    CudaCache *Cache = init_cache(llama3_model);
    char *input_str = strdup("The Pacific Ocean is the largest ocean in the\n");

    /*
    char *input_str = (char *)malloc(sizeof(char) * 2048);
    fgets(input_str, 2048, stdin);
    */

    int *tokens = tokenize(llama3_tokenizer, input_str);
    if (tokens == NULL) {
        printf("Error: Tokenization failed\n");
        return 1;
    }

    Tensor *X = (Tensor *)malloc(sizeof(Tensor));
    int *d_tokens = tokens_to_cuda(tokens, 4096, X);
    printf(GREEN "[CUDA]" RESET " Tokenized input and moved to CUDA Device\n");

    inference(llama3_model, X, d_tokens, tokens, Cache);

    // Free the model resources
    free_llama3(llama3_model);

    return 0;
}