#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama3/llama3.cuh"
#include "safetensor/safetensor.cuh"
#include "tokenizer/tokenizer.cuh"

#define WARN "\033[1;33m"
#define GREEN "\033[1;32m"
#define GREY "\033[2m"
#define RESET "\033[0m"

#define CLEAR_TERMINAL() system("clear")

const int MODEL_NUM_LAYERS = 32;

// CUDA kernel to check the 0th index of the fp16 tensor in the k_proj
__global__ void checker(__half *fp16_tensor, long *mem_len) {
    printf("The 0th index of the fp16_tensor (self_attn_k_proj): %f\n", __half2float(fp16_tensor[0]));
    printf("Mem Len: %lu\n", *mem_len);
}

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
    CLEAR_TERMINAL();

    printf(WARN "[CPU]" RESET " Loaded model to CPU\n");
    printf(WARN "[CPU]" RESET " Moving model to CUDA and converting model parameters from BF16 to FP16\n");

    to_cuda(llama3_model);
    bf16_to_fp16(llama3_model);
    printf(GREEN "[CUDA]" RESET " Loaded to CUDA Device and formatted Parameters to FP16\n");

    // Check the 0th index of the k_proj tensor of the first layer
    checker<<<1, 1>>>(
        llama3_model->embed_tokens->d_fp16_tensor, llama3_model->embed_tokens->d_mem_len);
    cudaDeviceSynchronize();

    // Load the tokenizer (this function should load the trie from the tokenizer's JSON)
    Llama3Tokenizer *llama3_tokenizer = load_tokenizer();
    if (llama3_tokenizer == NULL) {
        printf("Error: Failed to load the tokenizer\n");
        return 1;
    }

    char *input_str = strdup("If you are reading the data from the Internet instead, the same techniques can generally be used with the response you get from your HTTP API (it will be a file-like object); however, it is heavily recommended to use the third-party Requests library instead, which includes built-in support for JSON requests.");
    int *tokens = tokenize(llama3_tokenizer, input_str);
    if (tokens == NULL) {
        printf("Error: Tokenization failed\n");
        return 1;
    }

    printf("Number of Tokens: %d\n", tokens[0]);
    for (int i = 1; i < tokens[0]; i++) {
        printf("%d, ", tokens[i]);
    }
    printf("\n");

    // Free the model resources
    free_llama3(llama3_model);

    return 0;
}
