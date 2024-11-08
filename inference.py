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
input_text = "What is the largest ocean in the world?\n"
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
                        (q_proj): Linear(in_features=4096,
                         out_features=4096, bias=False)
                        (k_proj): Linear(in_features=4096,
                         out_features=1024, bias=False)
                        (v_proj): Linear(in_features=4096,
                         out_features=1024, bias=False)
                        (o_proj): Linear(in_features=4096,
                         out_features=4096, bias=False)
                        (rotary_emb): LlamaRotaryEmbedding()
                    )
                    (mlp): LlamaMLP(
                        (gate_proj): Linear(in_features=4096,
                         out_features=14336, bias=False)
                        (up_proj): Linear(in_features=4096,
                         out_features=14336, bias=False)
                        (down_proj): Linear(in_features=14336,
                         out_features=4096, bias=False)
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
    def rotate_half(x):
        """Rotate pairs of dimensions for rotary embedding."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def rotary_embedding(tensor, seq_len, embed_dim):
        # Compute frequency base for each pair of embedding dimensions
        half_dim = embed_dim // 2
        freqs = torch.pow(500000, -torch.arange(0, half_dim,
                                                dtype=torch.float32) / half_dim).to(device)

        # Create position indices
        position_ids = torch.arange(seq_len, dtype=torch.float32).to(device)

        # Create sinusoidal embedding (seq_len, half_dim)
        sinusoid_inp = torch.einsum("i,j->ij", position_ids, freqs)
        sin_embed = torch.sin(sinusoid_inp).to(device)
        cos_embed = torch.cos(sinusoid_inp).to(device)

        # Repeat to match input dimensions (tokens, embed_dim)
        sin_embed = sin_embed.repeat_interleave(
            2, dim=-1).to(device)
        cos_embed = cos_embed.repeat_interleave(
            2, dim=-1).to(device)

        # Apply the rotation to the tensor
        tensor_rotated = (tensor * cos_embed) + \
            (rotate_half(tensor) * sin_embed)

        return tensor_rotated.to(device)

    nheads = 32
    embed_dim = 4096
    head_dim = embed_dim // nheads
    seq_len = X.shape[-1]
    X = model.model.embed_tokens(X).half()

    for i in range(0, 32):
        LAYER = model.model.layers[i]

        PN_X = torch.clone(X)
        X = LAYER.input_layernorm(X)

        Q = LAYER.self_attn.q_proj(X).view(-1, seq_len, nheads, head_dim)
        K = LAYER.self_attn.k_proj(X).view(-1, seq_len, nheads // 4, head_dim)
        V = LAYER.self_attn.v_proj(X).view(-1, seq_len, nheads // 4, head_dim)

        # Shape: [batch_size, nheads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        # Shape: [batch_size, n_kv_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        # Shape: [batch_size, n_kv_heads, seq_len, head_dim]
        V = V.transpose(1, 2)

        K = K.repeat_interleave(4, dim=1)
        V = V.repeat_interleave(4, dim=1)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)

        # Apply causal mask (if necessary)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
        # Shape: [1, 1, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        print(attn_probs.shape)
        exit(1)

        # Apply attention to values
        Attention = torch.matmul(attn_probs, V)

        # Transpose back
        Attention = Attention.transpose(
            1, 2).contiguous().view(-1, seq_len, embed_dim)

        X = LAYER.self_attn.o_proj(Attention)
        X = X + PN_X

        PN_X = torch.clone(X)
        X = LAYER.post_attention_layernorm(X)

        GATE = LAYER.mlp.act_fn(LAYER.mlp.gate_proj(X))
        UP = LAYER.mlp.up_proj(X)

        X = GATE * UP
        X = LAYER.mlp.down_proj(X)

        X = X + PN_X

    X = model.model.norm(X)
    X = model.lm_head(X).reshape(seq_len, -1)

    print(torch.argmax(F.softmax(X, dim=-1), dim=-1))
