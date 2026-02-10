### Notes

TODOs
- checkpointing
- flops appx (for determing steps)
- optimizers (build own, dont use torch's)
- batch lr scaling


First, tokenize:
1. uv run python -m gptest.data.dataset -n 100 -w 16
2. uv run python -m gptest.tokenizer.tok_train


