import numpy as np
import json
import regex as re
import time

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # inverse vocab
        self.bytes_to_ids = {byte: id for id, byte in self.vocab.items()}
        self.merge_rank = {merge: i for i, merge in enumerate(self.merges)}
        self.merge_cache = {}

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.PAT_compiled = re.compile(PAT)

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            self.special_token_pattern = re.compile("(" + "|".join(escaped_tokens) + ")")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_json = json.load(vf)
            vocab = {int(k): bytes(v) for k, v in vocab_json.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                left, right = line.strip().split("\t")
                b1 = bytes(int(b) for b in left.split())
                b2 = bytes(int(b) for b in right.split())
                merges.append((b1, b2))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def apply_merges(self, pre_token):
        # repeatedly traverse the token for candidate merges,
        # choose the one with the highest priority,
        # merge it, and repeat
        key = tuple(pre_token)
        if key in self.merge_cache:
            return self.merge_cache[key]

        while True:
            merge_candidates = []

            # cache all of the pairs in the token
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i+1])
                if pair in self.merge_rank:
                    merge_candidates.append((self.merge_rank[pair], i))

            if not merge_candidates:
                break

            # highest priority + earliest in the token
            index = min(merge_candidates)[1]
            pre_token[index] = pre_token[index] + pre_token[index+1]
            del pre_token[index+1]

        self.merge_cache[key] = pre_token
        return pre_token

    def tokens_to_ids(self, tokens: list[bytes]) -> list[int]:
        return [self.bytes_to_ids[token] for token in tokens]

    def encode(self, text, max_chunks=None):
        def process_chunk(text):
            token_ids = []
            for match in re.finditer(self.PAT_compiled, text):
                # tuple of single bytes, each is a bytes type
                raw_bytes = match.group(0).encode("utf-8")
                pre_token = [bytes([b]) for b in raw_bytes]
                merged_segment = self.apply_merges(pre_token)
                token_ids.extend(self.tokens_to_ids(merged_segment))

            return token_ids

        # special tokens
        if self.special_tokens is None or len(self.special_tokens) == 0:
            return process_chunk(text)
        else:
            token_ids = []
            num_chunks = 0
            total_bytes = 0
            start_time = time.time()
            segments = re.split(self.special_token_pattern, text)
            # print("Total chunks: ", len(segments))
            for segment in segments:
                if self.special_tokens and segment in self.special_tokens:
                    token_ids.extend(self.tokens_to_ids([segment.encode("utf-8")]))
                else:
                    token_ids.extend(process_chunk(segment))
                num_chunks += 1

                total_bytes += len(segment)
                if max_chunks and num_chunks >= max_chunks:
                    break

                if num_chunks % 100000 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    mb = total_bytes / (1024 ** 2)
                    elapsed_minutes = elapsed / 60
                    throughput = mb / elapsed
                    # print(f"num_chunks: {num_chunks}")
                    # print(f"total_bytes: {total_bytes}")
                    # print(f"throughput: {throughput:.2f} MB/sec")
                    # print(f"elapsed: {elapsed_minutes:.2f} minutes")

            end_time = time.time()
            elapsed = end_time - start_time
            # print(f"elapsed: {elapsed}")
            # print(f"throughput: {total_bytes / elapsed:.2f} bytes/sec")

            # calculate compression ratio
            original_size = total_bytes
            encoded_size = len(token_ids) * 2
            compression_ratio = original_size / encoded_size
            # print(f"Original size: {original_size}")
            # print(f"Encoded size: {encoded_size}")
            # print(f"Compression ratio: {compression_ratio:.2f}")

        return token_ids

    def serialize(self, token_ids, token_ids_path):
        # we have 32K vocab size so uint16 works
        token_ids_array = np.array(token_ids, dtype=np.uint16)
        np.save(token_ids_path, token_ids_array)

    def encode_iterable(self, iterable):
        # iterable of strings (python file handle) -> generator that lazily yields token IDs
        for text in iterable:
            token_ids = self.encode(text)
            for token in token_ids:
                yield token

    def decode(self, ids):
        # Decode a list of token IDs into text
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_sequence.decode("utf-8", errors="replace")
