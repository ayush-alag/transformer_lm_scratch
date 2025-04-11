import regex as re
from collections import defaultdict
import heapq
import multiprocessing as mp
from functools import partial
import json

from .common_tokenizer import find_chunk_boundaries


class PairEntry:
    def __init__(self, pair, count):
        self.pair = pair  # The byte pair
        self.count = count  # Frequency count

    def __lt__(self, other):
        # For heap operations, we need to define "less than"
        # We want higher counts to come first (max heap)
        if self.count != other.count:
            return self.count > other.count  # Note the > for max heap

        # Tiebreaker: lexicographically greater pair comes first
        return self.pair > other.pair

    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair

    def __repr__(self):
        return f"PairEntry({self.pair}, {self.count})"


class BPETrainer:
    def __init__(self, vocab_size, input_path, special_tokens):
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.special_tokens = special_tokens

    def pretokenize_chunk(self, args, file_path, special_token_pattern):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        bytes_to_count = defaultdict(int)

        start, end = args
        # Each process needs its own file handle
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            segments = [chunk]
            if special_token_pattern:
                segments = special_token_pattern.split(chunk)

            for segment in segments:
                if segment in self.special_tokens:
                    token_bytes = segment.encode("utf-8")
                    # Wrap it in a tuple form (or do whatever format you use)
                    token_tuple = (token_bytes,)
                    bytes_to_count[token_tuple] += 1
                else:
                    for match in re.finditer(PAT, segment):
                        # tuple of single bytes, each is a bytes type
                        raw_bytes = match.group(0).encode("utf-8")
                        token_bytes = tuple(bytes([b]) for b in raw_bytes)
                        bytes_to_count[token_bytes] += 1

        return bytes_to_count

    def pretokenize_input(self):
        bytes_to_count = defaultdict(int)
        pair_to_locations = defaultdict(set)

        # Create a regex pattern to split on special tokens if provided
        special_token_pattern = None
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        if len(escaped_tokens) > 0:
            special_token_pattern = re.compile("(" + "|".join(escaped_tokens) + ")")

        # TODO: change this
        num_processes = 30

        ## Usage
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, "<|endoftext|>".encode("utf-8")
            )

            # Create argument pairs
            chunk_args = list(zip(boundaries[:-1], boundaries[1:]))

            # Create a partial function with fixed arguments
            worker_fn = partial(
                self.pretokenize_chunk,
                file_path=self.input_path,
                special_token_pattern=special_token_pattern,
            )

            # Create a process pool and map the work
            with mp.Pool(processes=mp.cpu_count()) as pool:
                # Process all chunks in parallel and get results
                results = pool.map(worker_fn, chunk_args)

            # Combine results from all processes
            for local_counts in results:
                for token, count in local_counts.items():
                    bytes_to_count[token] += count

        # build a unique token list that we maintain indices into
        unique_token_list = list(bytes_to_count.keys())
        pair_to_count = defaultdict(int)
        for i, unique_token in enumerate(unique_token_list):
            # Index all pairs in this token
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
        pair_to_count,
    ):
        # print("BEST PAIR: ", best_pair, "\n")
        # print("BEFORE: ", pair_to_count)
        merged_pair = best_pair[0] + best_pair[1]

        # collect all old tokens to delete at the end
        tokens_to_delete = set()

        # now, all i need to do is update pair_to_count, pair_to_locs, and pretoken
        for token_list_idx in list(pair_to_locs[best_pair]):
            old_token = unique_token_list[token_list_idx]

            # how many times this pretoken appears in the corpus
            pretoken_count = pretokenized_bytes_to_count[old_token]
            # print("OLD TOKEN: ", old_token, pretoken_count)

            # remove all pair counts for the old token
            for j in range(len(old_token) - 1):
                pair = (old_token[j], old_token[j + 1])
                pair_to_count[pair] -= pretoken_count
                pair_to_locs[pair].discard(token_list_idx)
                if pair_to_count[pair] <= 0:
                    del pair_to_count[pair]
                    del pair_to_locs[pair]

            # i corresponds to indexing into old token
            new_token = []
            i = 0
            while i < len(old_token):
                if (
                    old_token[i] == best_pair[0]
                    and i < len(old_token) - 1
                    and old_token[i + 1] == best_pair[1]
                ):
                    new_token.append(merged_pair)
                    i += 2
                else:
                    new_token.append(old_token[i])
                    i += 1
            new_token = tuple(new_token)

            # update counts
            unique_token_list[token_list_idx] = new_token
            pretokenized_bytes_to_count[new_token] += pretoken_count
            # print("NEW TOKEN: ", new_token, pretokenized_bytes_to_count[new_token])

            # Recompute and add back the count for every adjacent pair in the new token.
            for k in range(len(new_token) - 1):
                pair = (new_token[k], new_token[k + 1])
                pair_to_count[pair] += pretoken_count
                pair_to_locs[pair].add(token_list_idx)

            tokens_to_delete.add(old_token)

        # After processing all tokens that had the best_pair, delete all marked old tokens.
        for token in tokens_to_delete:
            pretokenized_bytes_to_count.pop(token, None)

        # After all token updates are complete, rebuild the heap
        pair_to_locs.pop(best_pair, None)
        pair_to_count.pop(best_pair, None)

        # print("AFTER: ", pair_to_count)

    def merge_tokens(
        self,
        pretokenized_bytes_to_count: dict[tuple[bytes], int],
        unique_token_list: list[bytes],
        pair_to_locs,
        pair_to_count,
        num_merges: int,
    ) -> list[tuple[bytes, bytes]]:
        merges = []

        pair_heap = [PairEntry(pair, count) for pair, count in pair_to_count.items()]
        heapq.heapify(pair_heap)

        for _ in range(num_merges):
            # get the max pair
            best_pair = heapq.heappop(pair_heap).pair

            # merge the max pair
            merges.append(best_pair)
            self.merge_pair(
                best_pair,
                pretokenized_bytes_to_count,
                unique_token_list,
                pair_to_locs,
                pair_to_count,
            )

            pair_heap = [
                PairEntry(pair, count)
                for pair, count in pair_to_count.items()
                if count > 0
            ]
            heapq.heapify(pair_heap)

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

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        pretokenized_bytes_to_count, pretoken_list, pair_to_locs, pair_to_count = (
            self.pretokenize_input()
        )

        number_merges = self.vocab_size - 256 - len(self.special_tokens)
        merges = self.merge_tokens(
            pretokenized_bytes_to_count,
            pretoken_list,
            pair_to_locs,
            pair_to_count,
            number_merges,
        )

        vocab = self.build_vocab(merges)

        return vocab, merges
