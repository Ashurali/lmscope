"""Char and BPE tokenizers with one small interface: encode, decode, token_str, vocab_size."""
import json
from pathlib import Path

from tokenizers import Tokenizer


class CharTokenizer:
    """One token per character, from a char_to_idx mapping."""

    kind = "char"

    def __init__(self, char_to_idx):
        self.stoi = {c: int(i) for c, i in char_to_idx.items()}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    @classmethod
    def from_file(cls, path):
        """JSON file holding either {"char_to_idx": {...}} or the mapping itself."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("char_to_idx", data))

    def encode(self, text):
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"characters not in the vocabulary: {unknown}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def token_str(self, i):
        return self.itos[int(i)]


class BPETokenizerWrapper:
    """HuggingFace `tokenizers` BPE model (the byte-level BPE trained in HW3)."""

    kind = "bpe"

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def from_file(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_str(cls, json_str):
        return cls(Tokenizer.from_str(json_str))

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids])

    def token_str(self, i):
        # Decoding one id on its own turns byte-level symbols (Ġ, Ċ) back into real spaces/newlines.
        return self.tok.decode([int(i)])


def load_tokenizer(tokenizer_type, path=None, checkpoint=None):
    """Loads from `path` if given, otherwise from the tokenizer info stored in the checkpoint."""
    checkpoint = checkpoint or {}
    tokenizer_type = tokenizer_type or checkpoint.get("tokenizer_type")
    if tokenizer_type == "char":
        if path:
            return CharTokenizer.from_file(path)
        if "char_to_idx" in checkpoint:
            return CharTokenizer(checkpoint["char_to_idx"])
        raise ValueError("char tokenizer: pass a tokenizer JSON file or use a checkpoint with char_to_idx")
    if tokenizer_type == "bpe":
        if path:
            return BPETokenizerWrapper.from_file(path)
        if "bpe_tokenizer_json" in checkpoint:
            return BPETokenizerWrapper.from_str(checkpoint["bpe_tokenizer_json"])
        raise ValueError("bpe tokenizer: pass the tokenizer JSON file (e.g. results/bpe2000_tokenizer.json)")
    raise ValueError(f"unknown tokenizer_type {tokenizer_type!r}")
