#pragma once

#include <stdint.h>
#include <stdio.h>

#include "Llama3/Llama3.cuh"
#include "cJSON/cJSON.h"

typedef struct {
    FILE *fp;
    char *header;
    uint64_t header_size;
} SafeTensorFile;

void load_safetensor_weights(Llama3 *llama3_model, const char *filename);

void safetensor_load_header(SafeTensorFile *STF);

void safetensor_read_header(SafeTensorFile *STF, Llama3 *llama3_model);

void llama3_load_layer(cJSON *json, SafeTensorFile *STF, Llama3 *llama3_model);

int get_llama3_decoder_layer_num(char *layer_key, int index);

void free_safetensor_handler(SafeTensorFile *STF);
