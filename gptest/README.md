### Notes

TODOs
- checkpointing
- flops appx
- optimizers (build own, dont use torch's)
- custom learning rate


First, tokenize:
1. uv run python -m gptest.data.dataset -n 100 -w 16
2. uv run python -m gptest.tokenizer.tok_train


