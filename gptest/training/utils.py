


def get_lr_multiplier(config, step, max_steps):
    warmup_iters = round(config.meta.warmup_ratio * max_steps)
    warmdown_iters = round(config.meta.warmdown_ratio * max_steps)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= max_steps - warmdown_iters:
        return 1.0
    else:
        progress = (max_steps - step) / warmdown_iters
        return progress + (1 - progress) * config.meta.final_lr_frac
