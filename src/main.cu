#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Llama3Weight/Llama3Weight.cuh"
#include "SafeTensor/SafeTensor.cuh"

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

    free_LLaMa3(llama3_model);

    return 0;
}