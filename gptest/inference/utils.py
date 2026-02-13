import torch
import torch.nn.functional as F


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    assert temperature >= 0, 'temperature must be >= 0'
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals /= temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits /= temperature
        probs = F.softma(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)
