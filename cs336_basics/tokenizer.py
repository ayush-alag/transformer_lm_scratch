import regex as re
from collections import defaultdict
import heapq

# TODO: need to add parallelism??
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
class BPETokenizer:
   def __init__(self, vocab_size, input_path, special_tokens):
      self.vocab_size = vocab_size
      self.input_path = input_path
      self.special_tokens = special_tokens
      
      # initialize vocabulary
      self.vocab = {i: bytes([i]) for i in range(256)}
      self.next_vocab_idx = 256

   def pretokenize_input(self) -> dict[tuple[bytes], int]:
      PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
      
      bytes_to_count = defaultdict(int)
      pretoken_list = []
      pair_to_locations = defaultdict(list)
      
      # Create a regex pattern to split on special tokens if provided
      special_token_pattern = None
      escaped_tokens = [re.escape(token) for token in self.special_tokens]
      if len(escaped_tokens) > 0:
         special_token_pattern = re.compile('|'.join(escaped_tokens))
      
      with open(self.input_path, 'r', encoding='utf-8') as f:
         for line in f:
            segments = [line]
            if special_token_pattern:
                  segments = special_token_pattern.split(line)
            
            for segment in segments:
               for match in re.finditer(PAT, segment):
                  # tuple of single bytes, each is a bytes type
                  token_bytes = tuple(match.group(0).encode('utf-8'))
                  bytes_to_count[token_bytes] += 1
                  
                  pretoken_list.append(token_bytes)
                  
                  # Index all pairs in this token
                  for pos in range(len(token_bytes) - 1):
                     pair = (token_bytes[pos], token_bytes[pos + 1])
                     pair_to_locations[pair].add((len(pretoken_list), pos))
      
      return bytes_to_count, pretoken_list, pair_to_locations

   def merge_tokens(self,
                    pretokenized_bytes_to_count: dict[tuple[bytes], int],
                    pretoken_list: list[bytes],
                    pair_to_locs,
                    pair_to_count,
                    num_merges: int) -> \
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
      merges = []
      
      pair_heap = [PairEntry(pair, count) for pair, count in pair_to_count.items()]
      heapq.heapify(pair_heap)
      
      for _ in range(num_merges):
         # get the max pair
         best_entry = heapq.heappop(pair_heap)
         best_pair = best_entry.pair
         best_count = best_entry.count
         
         # merge the max pair
         merges.append(best_pair)
         self.add_to_vocab(best_pair[0] + best_pair[1])
         
         locs = pair_to_locs[best_pair]
         # do something with these locations and pretoken_list
         
         
      
      return merges

   def add_to_vocab(self, encoded_token):
      self.vocab[self.next_vocab_idx] = encoded_token
      self.next_vocab_idx += 1
      
   def add_special_tokens(self):
      for token in self.special_tokens:
         self.add_to_vocab(token.encode('utf-8'))

   def train_bpe(self) -> \
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
         pretokenized_bytes_to_count, pretoken_list, pair_to_locs = self.pretokenize_input()
         pair_to_count = {pair : len(locations) for pair, locations in pair_to_locs.items()}
         
         number_merges = self.vocab_size - len(self.vocab) - len(self.special_tokens)
         merges = self.merge_tokens(pretokenized_bytes_to_count,
                                    pretoken_list,
                                    pair_to_locs,
                                    pair_to_count,
                                    number_merges)

         self.add_special_tokens()
         
         return self.vocab, merges
         