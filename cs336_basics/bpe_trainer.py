import regex as re
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import json
import cProfile
import pstats
from pstats import SortKey
import time
import mmap
import psutil
import os
import heapdict

from .common_tokenizer import find_chunk_boundaries

class PairEntry:
    def __init__(self, pair, count):
        self.pair = pair
        self.count = count

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count

        return self.pair > other.pair

    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair

    def __repr__(self):
        return f"PairEntry({self.pair}, {self.count})"

class BPETrainer:
    def __init__(self, vocab_size, input_path, special_tokens, num_processes=1):
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.num_processes = num_processes

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.PAT_compiled = re.compile(PAT)

    def pretokenize_chunk(self, args, file_path, special_token_pattern=None):
        bytes_to_count = defaultdict(int)

        start, end = args
        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            chunk = mm[start:end].decode("utf-8", errors="ignore")
            mm.close()

            if special_token_pattern:
                for small_chunk in re.split(special_token_pattern, chunk):
                    for match in re.finditer(self.PAT_compiled, small_chunk):
                        # tuple of single bytes, each is a bytes type
                        raw_bytes = match.group(0).encode("utf-8")
                        token_bytes = tuple(bytes([b]) for b in raw_bytes)
                        bytes_to_count[token_bytes] += 1
            else:
                for match in re.finditer(self.PAT_compiled, chunk):
                    raw_bytes = match.group(0).encode("utf-8")
                    token_bytes = tuple(bytes([b]) for b in raw_bytes)
                    bytes_to_count[token_bytes] += 1

        return bytes_to_count

    def pretokenize_input(self):
        bytes_to_count = defaultdict(int)
        pair_to_locations = defaultdict(set)

        special_token_pattern = None
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        if len(escaped_tokens) > 0:
            special_token_pattern = re.compile("|".join(escaped_tokens))

        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, self.num_processes, "<|endoftext|>".encode("utf-8")
            )

            chunk_args = list(zip(boundaries[:-1], boundaries[1:]))

            # partial function with fixed arguments
            worker_fn = partial(
                self.pretokenize_chunk,
                file_path=self.input_path,
                special_token_pattern=special_token_pattern,
            )

            # print(f"Pretokenizing with {self.num_processes} processes...")
            with mp.Pool(processes=self.num_processes) as pool:
                results = pool.map(worker_fn, chunk_args)

            for local_counts in results:
                for token, count in local_counts.items():
                    bytes_to_count[token] += count

        # build a unique token list that we maintain indices into
        unique_token_list = list(bytes_to_count.keys())
        pair_to_count = defaultdict(int)
        for i, unique_token in enumerate(unique_token_list):
            # index all pairs in this token
            for pos in range(len(unique_token) - 1):
                pair = (unique_token[pos], unique_token[pos + 1])
                pair_to_locations[pair].add(i)
                pair_to_count[pair] += bytes_to_count[unique_token]

        return bytes_to_count, unique_token_list, pair_to_locations, pair_to_count

    def merge_pair(
        self,
        best_pair,
        pretokenized_bytes_to_count: dict[tuple[bytes], int],
        unique_token_list: list[bytes],
        pair_to_locs,
        pair_to_count
    ):
        merged_pair = best_pair[0] + best_pair[1]
        tokens_to_delete = set()
        affected_pairs = set()

        # now, all i need to do is update pair_to_count, pair_to_locs, and pretoken
        for token_list_idx in list(pair_to_locs[best_pair]):
            old_token = unique_token_list[token_list_idx]

            # how many times this pretoken appears in the corpus
            pretoken_count = pretokenized_bytes_to_count[old_token]

            # remove all pair counts for the old token
            for j in range(len(old_token) - 1):
                pair = (old_token[j], old_token[j + 1])
                pair_to_count[pair] -= pretoken_count
                pair_to_locs[pair].discard(token_list_idx)
                affected_pairs.add(pair)
                if pair_to_count[pair] <= 0:
                    del pair_to_count[pair]
                    del pair_to_locs[pair]

            # i corresponds to indexing into old token
            new_token = []
            i = 0
            while i < len(old_token):
                if old_token[i] == best_pair[0] and i < len(old_token) - 1 and old_token[i + 1] == best_pair[1]:
                    new_token.append(merged_pair)
                    i += 2
                else:
                    new_token.append(old_token[i])
                    i += 1
            new_token = tuple(new_token)

            # update counts
            unique_token_list[token_list_idx] = new_token
            pretokenized_bytes_to_count[new_token] += pretoken_count

            # Recompute and add back the count for every adjacent pair in the new token.
            for k in range(len(new_token) - 1):
                pair = (new_token[k], new_token[k + 1])
                pair_to_count[pair] += pretoken_count
                pair_to_locs[pair].add(token_list_idx)
                affected_pairs.add(pair)

            tokens_to_delete.add(old_token)

        # after processing all tokens that had the best_pair, delete all marked old tokens.
        for token in tokens_to_delete:
            pretokenized_bytes_to_count.pop(token, None)

        pair_to_locs.pop(best_pair, None)
        pair_to_count.pop(best_pair, None)

        return affected_pairs

    def lex_key(self,b: bytes):
        return tuple(-x for x in b) + (len(b),)

    def get_priority(self, pair, count):
        return (-count, self.lex_key(pair[0]), self.lex_key(pair[1]))

    def merge_tokens(
        self,
        pretokenized_bytes_to_count: dict[tuple[bytes], int],
        unique_token_list: list[bytes],
        pair_to_locs,
        pair_to_count,
        num_merges: int,
    ) -> list[tuple[bytes, bytes]]:
        merges = []

        hp = heapdict.heapdict()
        for pair, count in pair_to_count.items():
            if count > 0:
                hp[pair] = self.get_priority(pair, count)

        for _ in range(num_merges):
            get_best_pair_time = time.time()
            # get the max pair
            best_pair, _ = hp.popitem()
            get_best_pair_time = time.time() - get_best_pair_time
            # print(f"Get best pair time: {get_best_pair_time:.2f} seconds")

            # merge the max pair
            merges.append(best_pair)
            merge_time = time.time()
            affected_pairs = self.merge_pair(
                best_pair,
                pretokenized_bytes_to_count,
                unique_token_list,
                pair_to_locs,
                pair_to_count
            )
            merge_time = time.time() - merge_time
            # print(f"Merge time: {merge_time:.2f} seconds")

            # rebuild the heap
            for pair in affected_pairs:
                if pair in pair_to_count and pair_to_count[pair] > 0:
                    hp[pair] = self.get_priority(pair, pair_to_count[pair])
                else:
                    hp.pop(pair, None)

        return merges

    def build_vocab(self, merges):
        # initialize vocabulary
        vocab = {i: bytes([i]) for i in range(256)}

        def add_to_vocab(vocab, encoded_token):
            vocab[len(vocab)] = encoded_token

        # add merges
        for merge in merges:
            add_to_vocab(vocab, merge[0] + merge[1])

        # add special tokens
        for token in self.special_tokens:
            add_to_vocab(vocab, token.encode("utf-8"))

        return vocab

    # @profile
    def train_bpe(self, profile_path=None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        start_total = time.time()
        if profile_path:
            profiler = cProfile.Profile()
            profiler.enable()

        start_pretokenize = time.time()
        pretokenized_bytes_to_count, pretoken_list, pair_to_locs, pair_to_count = (
            self.pretokenize_input()
        )
        end_pretokenize = time.time()
        pretokenize_time = end_pretokenize - start_pretokenize
        # print(f"Pretokenization completed in {pretokenize_time:.2f} seconds")

        start_merge = time.time()
        number_merges = self.vocab_size - 256 - len(self.special_tokens)
        merges = self.merge_tokens(
            pretokenized_bytes_to_count,
            pretoken_list,
            pair_to_locs,
            pair_to_count,
            number_merges,
        )
        # end_merge = time.time()
        # merge_time = end_merge - start_merge
        # # print(f"Merge operations completed in {merge_time:.2f} seconds")

        # start_vocab = time.time()
        vocab = self.build_vocab(merges)
        # end_vocab = time.time()
        # vocab_time = end_vocab - start_vocab
        # # print(f"Vocabulary building completed in {vocab_time:.2f} seconds")

        # # Print total time
        # end_total = time.time()
        # total_time = end_total - start_total
        # print(f"\nTotal execution time: {total_time:.2f} seconds")
        # print(f"  - Pretokenization: {pretokenize_time:.2f}s ({pretokenize_time/total_time*100:.1f}%)")
        # print(f"  - Merge operations: {merge_time:.2f}s ({merge_time/total_time*100:.1f}%)")
        # print(f"  - Vocabulary building: {vocab_time:.2f}s ({vocab_time/total_time*100:.1f}%)")
        # print(f"  - Other overhead: {total_time - (pretokenize_time + merge_time + vocab_time):.2f}s")

        if profile_path:
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(50)
            stats.dump_stats(profile_path)

        return vocab, merges

# serialize to use later
def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
):
    # Save vocab as JSON: {int: list of ints (byte values)}
    with open(vocab_path, "w", encoding="utf-8") as vf:
        json.dump({k: list(v) for k, v in vocab.items()}, vf, indent=2)

    # Save merges as space-separated byte values (ints)
    with open(merges_path, "w", encoding="utf-8") as mf:
        for b1, b2 in merges:
            mf.write(
                f"{' '.join(str(b) for b in b1)}\t{' '.join(str(b) for b in b2)}\n"
            )

def report_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # Resident Set Size in bytes
    # print(f"Memory usage (RSS): {mem / (1024 * 1024):.2f} MB")
