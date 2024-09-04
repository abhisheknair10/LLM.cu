# LLM.cu - A LLaMa3-8B CUDA Inference Engine

LLM.cu is a CUDA native implementation of the LLaMa3 architecture for sequence to sequence language modeling. Core principles of the transformer architecture from the papers [Attention is All You Need](https://arxiv.org/abs/1706.03762) and [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) are implemented using custom CUDA kernel definitions, facilitating scalable parallel processing on Nvidia GPUs.

## Implementation Details

## Usage

```bash
# Install the model weights 
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./model_weights/ --token $HF_TOKEN
```