# lmscope

Step through token-by-token generation of a small LSTM or Transformer language model in the browser,
and watch the probability distribution, attention maps (Transformer) or gate activations (LSTM).

```bash
pip install lmscope       # from PyPI; or, from a clone: pip install -e .
lmscope serve --model transformer_char_best.pt --model-type transformer --tokenizer-type char
lmscope serve --model lstm_bpe2000_best.pt --model-type lstm \
              --tokenizer bpe2000_tokenizer.json --tokenizer-type bpe
```

```python
import lmscope
lmscope.serve(model_path="transformer_char_best.pt", model_type="transformer", tokenizer_type="char", port=8765)
```

Options: `--prompt "ROMEO:\n"`, `--seed 0` (sampling seed, restored on Reset), `--reference valid.txt`
(preloads a reference continuation for per-token loss / perplexity), `--device cpu`, `--no-browser`.

## Checkpoint format

```python
{
    "model_state_dict": ...,
    "model_config": {"model_type": "transformer" | "lstm", "vocab_size": int, "d_model": int,
                     "n_layers": int, "context_len": int, "dropout": float,
                     "n_heads": int, "d_ff": int,          # Transformer
                     "hidden_size": int},                  # LSTM
    "tokenizer_type": "char" | "bpe",
    "char_to_idx": {...}, "idx_to_char": {...},            # char
    "bpe_tokenizer_json": str,                             # bpe (optional if --tokenizer is given)
}
```

The tokenizer comes from `--tokenizer` if given, otherwise from the checkpoint.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /api/model-info` | | model type, parameter count, layers, vocab size, context length |
| `GET /api/state` | | token sequence with per-token sampling probability and loss |
| `POST /api/set-prompt` | `{prompt}` | tokenized prompt (resets the sequence) |
| `POST /api/reset` | | back to the prompt; sampling seed restored |
| `POST /api/step` | `{position, temperature, layer, head, top_k}` | one forward pass: top-k distribution, entropy, top-k mass, attention matrix or LSTM gates, hidden-state norms |
| `POST /api/sample` | `{temperature}` | samples and appends the next token |
| `POST /api/score` | `{text}` | per-token cross-entropy and running perplexity of a reference continuation |

`position` p is the prediction of token p from the tokens before it (default: the next token).

## Internals

- **Transformer**: each attention module applies its softmax through a small `AttentionSoftmax`
  submodule; a forward hook on it records the weights `(heads, T, T)` of every layer.
- **LSTM**: `CustomLSTMCell` re-runs each `nn.LSTM` layer step by step from its weights
  (gate order input, forget, cell, output) and exposes the input / forget / output gates and the states.

PNG export happens in the browser: per-panel buttons save each canvas, and "Export page" uses
html2canvas (loaded from cdnjs, so it needs internet access).

## Releasing

The version lives in `lmscope/__init__.py` (`__version__`).

1. Bump `__version__`, commit and push.
2. Create a GitHub release with tag `vX.Y.Z`. The `Publish to PyPI` workflow builds the sdist and wheel,
   runs `twine check`, and uploads with PyPI trusted publishing (no API token in the repo).

One-time setup on PyPI: under *Your projects → Publishing → Add a pending publisher*, enter owner `Ashurali`,
repository `lmscope`, workflow `publish.yml`, environment `pypi`.

Local build check: `python -m build && python -m twine check dist/*`.
