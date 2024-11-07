from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

# Load the tokenizer and model from Hugging Face's model hub
# Replace with the exact path to the LLaMA 3 model
model_name = "./model_weights/"

# Check if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)

# Define input text
input_text = "The largest ocean in the world is the"
X = tokenizer(input_text, return_tensors="pt").input_ids.to(device)


def SMART_PRINT(tensor):
    print(tensor.shape)
    print(tensor)


# Generate output
with torch.no_grad():
    """
    LlamaForCausalLM(
        (model): LlamaModel(
            (embed_tokens): Embedding(128256, 4096)
            (layers): ModuleList(
                (0-31): 32 x LlamaDecoderLayer(
                        (self_attn): LlamaSdpaAttention(
                        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                        (rotary_emb): LlamaRotaryEmbedding()
                    )
                    (mlp): LlamaMLP(
                        (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
                        (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
                        (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
                        (act_fn): SiLU()
                    )
                    (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
                    (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
                )
            )
            (norm): LlamaRMSNorm((4096,), eps=1e-05)
            (rotary_emb): LlamaRotaryEmbedding()
        )
        (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
    )
    """
    nheads = 32
    embed_dim = 4096
    head_dim = (int)(nheads / embed_dim)
    X = model.model.embed_tokens(X).half()

    for i in range(0, 32):
        LAYER = model.model.layers[i]

        PN_X = torch.clone(X)
        X = LAYER.input_layernorm(X)

        Q = LAYER.rotary_emb(LAYER.self_attn.q_proj(
            X)).reshape(nheads, -1, head_dim)
        K = LAYER.rotary_emb(LAYER.self_attn.k_proj(
            X)).reshape(nheads / 4, -1, head_dim)
        V = LAYER.self_attn.v_proj(
            X).reshape(nheads / 4, -1, head_dim)

        Attention = torch.matmul(
            F.softmax(torch.matmul(Q, K.t()) / (head_dim ** 0.5), 1),
            V
        )

        X = X.reshape(-1, embed_dim)
        X = LAYER.self_attn.o_proj(X)
        X = X + PN_X

        PN_X = torch.clone(X)
        X = LAYER.post_attention_layernorm(X)

        GATE = LAYER.mlp.act_fn(LAYER.mlp.gate_proj(X))
        UP = LAYER.mlp.up_proj(X)

        X = GATE * UP
        DOWN = LAYER.mlp.gate_proj(X)

        X = X + PN_X

    X = model.model.norm(X)
    X = model.model.lm_head(X)

    print(torch.argmax(X, 1))
