from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

# Load the tokenizer and model from Hugging Face's model hub
model_name = "./model_weights/"

# Check if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
model.eval()  # Set model to evaluation mode


def rotate_half(x):
    """
    Rotate half of the hidden dimensions of the input tensor.

    Args:
    x: Query or Key tensor of shape [batch_size, nheads, seq_len, head_dim]

    Returns:
    Rotated tensor.
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
    q, k: Query and Key tensors of shape [batch_size, nheads, seq_len, head_dim]
    cos, sin: Cosine and sine tensors of shape [batch_size, 1, seq_len, head_dim]

    Returns:
    q_embed, k_embed: Query and Key tensors with rotary embeddings applied.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def generate_token(model, X):
    """
    Generate the next token using the model by manually implementing the forward pass.

    Args:
    model: The causal language model.
    X: Input token IDs tensor of shape [batch_size, seq_len]

    Returns:
    Next token ID tensor of shape [batch_size]
    """
    with torch.no_grad():
        nheads = 32
        embed_dim = 4096
        head_dim = embed_dim // nheads  # 128
        seq_len = X.shape[-1]
        batch_size = X.shape[0]

        # Embed tokens
        X = model.model.embed_tokens(X)  # [B, S, 4096]

        for i in range(32):
            LAYER = model.model.layers[i]

            PN_X = X  # Residual connection

            # Layer normalization
            X = LAYER.input_layernorm(X)

            # Project to Q, K, V
            Q = LAYER.self_attn.q_proj(X).view(
                batch_size, seq_len, nheads, head_dim)  # [B, S, 32, 128]
            K = LAYER.self_attn.k_proj(X).view(
                batch_size, seq_len, nheads // 4, head_dim)  # [B, S, 8, 128]
            V = LAYER.self_attn.v_proj(X).view(
                batch_size, seq_len, nheads // 4, head_dim)  # [B, S, 8, 128]

            # Transpose for attention computation
            Q = Q.transpose(1, 2)  # [B, 32, S, 128]
            K = K.transpose(1, 2)  # [B, 8, S, 128]
            V = V.transpose(1, 2)  # [B, 8, S, 128]

            # Apply Rotary Positional Embeddings
            position_ids = torch.arange(seq_len, device=device).unsqueeze(
                0).expand(batch_size, -1)  # [B, S]
            cos = LAYER.self_attn.rotary_emb.cos[position_ids].unsqueeze(
                1)  # [B, 1, S, 128]
            sin = LAYER.self_attn.rotary_emb.sin[position_ids].unsqueeze(
                1)  # [B, 1, S, 128]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

            # Broadcast K and V to match Q's heads without duplication
            K = K.unsqueeze(2).expand(-1, -1, 4, -1, -1).reshape(batch_size,
                                                                 nheads, seq_len, head_dim)  # [B, 32, S, 128]
            V = V.unsqueeze(2).expand(-1, -1, 4, -1, -1).reshape(batch_size,
                                                                 nheads, seq_len, head_dim)  # [B, 32, S, 128]

            # Compute attention scores
            attn_scores = torch.matmul(
                Q, K.transpose(-1, -2)) / (head_dim ** 0.5)  # [B, 32, S, S]

            # Create and apply causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len,
                              device=device, dtype=attn_scores.dtype))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

            # Compute attention probabilities
            attn_probs = F.softmax(attn_scores, dim=-1)  # [B, 32, S, S]

            # Apply attention to values
            Attention = torch.matmul(attn_probs, V)  # [B, 32, S, 128]

            # Reshape Attention
            Attention = Attention.transpose(1, 2).contiguous().view(
                batch_size, seq_len, embed_dim)  # [B, S, 4096]

            # Output projection
            X = LAYER.self_attn.o_proj(Attention)  # [B, S, 4096]
            X = X + PN_X  # Residual connection

            # Post-attention LayerNorm and MLP
            PN_X = X  # Residual
            X = LAYER.post_attention_layernorm(X)

            GATE = F.silu(LAYER.mlp.gate_proj(X))  # [B, S, 14336]
            UP = LAYER.mlp.up_proj(X)  # [B, S, 14336]
            X = LAYER.mlp.down_proj(GATE * UP)  # [B, S, 4096]

            X = X + PN_X  # Residual connection

        # Final LayerNorm and Language Modeling Head
        X = model.model.norm(X)  # [B, S, 4096]
        X = model.lm_head(X).reshape(
            batch_size, seq_len, -1)  # [B, S, vocab_size]

        # Generate the next token by selecting the highest probability token
        next_token = torch.argmax(X, dim=-1)  # [B, S]

        return next_token


input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant, here to provide clear and concise answers to the user's questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the largest ocean in the world?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
max_tokens = 100

for _ in range(max_tokens):
    # Tokenize input text
    X = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate next token
    tok = generate_token(model, X)[0, -1].item()

    # Decode the token
    token_actual = tokenizer.decode(tok)

    # Break condition based on vocabulary size
    if tok >= model.config.vocab_size:
        print("\nEncountered invalid token ID. Stopping generation.")
        break

    # Print the token
    print(token_actual, end="", flush=True)

    # Append the generated token to the input text
    input_text += token_actual

print()
