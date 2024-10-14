# LLM.cu - A LLaMA3-8B CUDA Inference Engine

LLM.cu is a CUDA native implementation of the LLaMA3 architecture for sequence to sequence language modeling. Core principles of the transformer architecture from the papers [Attention is All You Need](https://arxiv.org/abs/1706.03762) and [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) are implemented using custom CUDA kernel definitions, facilitating scalable parallel processing on Nvidia GPUs.

## Implementation Details

## Setup and Usage

1. Run the **[setup-docker.sh](https://github.com/abhisheknair10/LLM.cu/blob/main/setup-docker.sh)** file to setup your Virtual/Physical Machine to run Docker with access to Nvidia GPUs. Once the shell script has finished executing, make sure to log out of the terminal, and then log back in to run **[run-docker.sh](https://github.com/abhisheknair10/LLM.cu/blob/main/run-docker.sh)**.

```bash
# Setup Docker
chmod +x setup-docker.sh
./setup-docker.sh
```

```bash
# Restart terminal and run
chmod +x run-docker.sh
./run-docker.sh
```

2. For this inference engine to work, a SafeTensor formatted file(s) of the Llama3-8b model needs to be stored in the ./model_weights/ folder. Head to the [HuggingFace - meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B?text=My+name+is+Julien+and+I+like+to) repo to get access to the model. Additionally, [Generate a Hugging Face Token](https://huggingface.co/settings/tokens) so that the next step can successfully download the weights files.

3. Once the Docker container has started up, run the following command to store the Hugging Face token as an environment variable, replacing **<your_token>** with the token you generated.

```bash
export HF_TOKEN=<your_token>
```

4. Next, run the following command to download the model parameters into the target directory.

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./model_weights/ --token $HF_TOKEN
```

## Acknowledgments

- This project makes use of the [cJSON library](https://github.com/DaveGamble/cJSON), which is licensed under the MIT License.
