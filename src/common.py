import torch
import torch.nn.functional as F

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': torch.float8_e4m3fn,
}


# https://arxiv.org/pdf/2104.09864
def apply_rotary_emb(x, cos, sin):
    # x shape: (B, T, H, D)
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split x in half
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def precompute_rotary_embeddings(config, seq_len, head_dim, base=100_000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # stride channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # stride time steps
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # calculate rotation frequencies
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    dtype = DTYPE_MAP[config.gpt.rope_dtype]
    cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)

    # add batch and num heads, shape: (T, D) -> (B, T, H, D) 
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin


# https://arxiv.org/pdf/1910.07467
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))