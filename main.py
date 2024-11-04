from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
X = tokenizer(input_text, return_tensors="pt").input_ids.half().to(device)


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
    model.model.embed_tokens.weight = model.model.embed_tokens.weight.float()
    X = model.model.embed_tokens(X.float()).half()
    X = model.model.layers[0].input_layernorm(X)
    Q = model.model.layers[0].self_attn.q_proj(X)
    Q_m = torch.matmul(
        X,
        model.model.layers[0].self_attn.q_proj.weight
    )

    SMART_PRINT(Q)
    print(torch.allclose(Q_m, Q, atol=1e-2))
