import numpy as np
import json
from collections import defaultdict

from .common_tokenizer import find_chunk_boundaries
import regex as re
from functools import partial
import multiprocessing as mp


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # inverse vocab
        self.bytes_to_ids = {byte: id for id, byte in self.vocab.items()}

    # TODO test this... with serialize + deserialize
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_json = json.load(vf)
            vocab = {int(k): bytes(v) for k, v in vocab_json.items()}

        # Load merges
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                left, right = line.strip().split("\t")
                b1 = bytes(int(b) for b in left.split())
                b2 = bytes(int(b) for b in right.split())
                merges.append((b1, b2))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def apply_merges(self, pre_token):
        # Apply merges to pre-token (list of bytes)
        for first_byte, second_byte in self.merges:
            merged_token = []
            i = 0
            while i < len(pre_token):
                if (
                    pre_token[i] == first_byte
                    and i < len(pre_token) - 1
                    and pre_token[i + 1] == second_byte
                ):
                    merged_token.append(first_byte + second_byte)
                    i += 2
                else:
                    merged_token.append(pre_token[i])
                    i += 1

            pre_token = merged_token

        return pre_token

    def tokens_to_ids(self, tokens: list[bytes]) -> list[int]:
        return [self.bytes_to_ids[token] for token in tokens]

    # TODO: parallelize? maybe not necessary
    def encode(self, text):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # no special tokens
        if self.special_tokens is None or len(self.special_tokens) == 0:
            raw_bytes = text.encode("utf-8")
            pre_token = [bytes([b]) for b in raw_bytes]
            merged = self.apply_merges(pre_token)
            return self.tokens_to_ids(merged)

        # special tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
        special_token_pattern = re.compile("(" + "|".join(escaped_tokens) + ")")
        segments = special_token_pattern.split(text)

        token_ids = []
        for segment in segments:
            if self.special_tokens and segment in self.special_tokens:
                token_ids.extend(self.tokens_to_ids([segment.encode("utf-8")]))
            else:
                for match in re.finditer(PAT, segment):
                    # tuple of single bytes, each is a bytes type
                    raw_bytes = match.group(0).encode("utf-8")
                    pre_token = [bytes([b]) for b in raw_bytes]
                    merged_segment = self.apply_merges(pre_token)
                    token_ids.extend(self.tokens_to_ids(merged_segment))

        return token_ids

    def encode_iterable(self, iterable):
        # iterable of strings (python file handle) -> generator that lazily yields token IDs
        pass

    def decode(self, ids):
        # Decode a list of token IDs into text
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_sequence.decode("utf-8", errors="replace")
