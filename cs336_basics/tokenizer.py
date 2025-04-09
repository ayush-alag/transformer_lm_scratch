import regex as re
from collections import defaultdict

# TODO: need to add parallelism??
class BPETokenizer:
   def __init__(self, vocab_size, input_path, special_tokens):
      self.vocab_size = vocab_size
      self.input_path = input_path
      self.special_tokens = special_tokens
      
      # populated by initialize_vocabulary
      self.next_vocab_idx = None
      
   def initialize_vocabulary(self) -> dict[int, bytes]:
      # for each of 256 bytestrings create a dict mapping to int
      init_vocab = {i: bytes([i]) for i in range(256)}
      self.next_vocab_idx = 256
      return init_vocab

   def pretokenize_input(self) -> dict[tuple[bytes], int]:
      PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
      
      bytes_to_count = defaultdict(int)
      
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
      
      return bytes_to_count

   def get_pair_counts(self, bytes_to_count: dict[tuple[bytes], int]):
      pair_to_count = defaultdict(int)
      for bytes, count in bytes_to_count.items():
         if len(bytes) < 2:
            continue
         
         for i in range(len(bytes) - 1):
            pair_to_count[bytes[i] + bytes[i + 1]] += count
      
      return pair_to_count

   def merge_pair(self, bytes_to_count, pair, vocab):
      vocab.append()
      pass

   def merge_tokens(self,
                    pretokenized_bytes_to_count: dict[tuple[bytes], int],
                    num_merges: int,
                    init_vocab: dict[int, bytes]) -> \
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
      vocab = init_vocab # modify in place
      merges = []
      
      pair_counts = self.get_pair_counts(pretokenized_bytes_to_count)
      # TODO find the max pair
      # TODO merge the max pair
      
      for _ in range(num_merges - 1):
         pass
      return vocab, merges

   def add_special_tokens(self, vocab):
      for token in self.special_tokens:
         vocab[self.next_vocab_idx] = token.encode('utf-8')
         self.next_vocab_idx += 1

   def train_bpe(self) -> \
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
         initial_vocab = self.initialize_vocabulary()
         pretokenized_bytes_to_count = self.pretokenize_input()
         
         number_merges = self.vocab_size - len(initial_vocab) - len(self.special_tokens)
         vocab, merges = self.merge_tokens(pretokenized_bytes_to_count, number_merges, initial_vocab)
         self.add_special_tokens(vocab)
         
         return vocab, merges
         