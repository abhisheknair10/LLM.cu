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
const bool TEST = true;

__global__ void model_param_checker(__half *fp16_tensor, int *mem_len);
__global__ void tokens_checker(int *tokens);

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

    char *input_str = strdup("If you are reading the data from the Internet instead, the same techniques can generally be used with the response you get from your HTTP API (it will be a file-like object); however, it is heavily recommended to use the third-party Requests library instead, which includes built-in support for JSON requests.");

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

    if (TEST) {
        // Check the 0th index of the k_proj tensor of the first layer
        model_param_checker<<<1, 1>>>(
            llama3_model->layers[0]->mlp_down_proj->d_fp16_tensor, llama3_model->layers[0]->mlp_down_proj->d_mem_len);
        cudaDeviceSynchronize();

        // Check if tokens have been stored in CUDA
        tokens_checker<<<1, 1>>>(d_tokens);
        cudaDeviceSynchronize();
    }

    // Free the model resources
    free_llama3(llama3_model);

    return 0;
}

// CUDA kernel to check the 0th index of the fp16 tensor in the k_proj
__global__ void model_param_checker(__half *fp16_tensor, int *mem_len) {
    printf("The 0th index of the fp16_tensor (mlp_down_proj): %f\n", __half2float(fp16_tensor[0]));
    printf("Mem Len: %d\n", *mem_len);
}

// CUDA kernel to check the tokens
__global__ void tokens_checker(int *tokens) {
    printf("Number of Tokens: %d\n", tokens[0] - 1);
    for (int i = 1; i < tokens[0]; i++) {
        printf("%d, ", tokens[i]);
    }
    printf("\n");
}