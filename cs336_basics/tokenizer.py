import regex as re
from collections import defaultdict

# TODO: need to add parallelism??
def initialize_vocabulary() -> dict[int, bytes]:
   # for each of 256 bytestrings create a dict mapping to int
   return {i: bytes([i]) for i in range(256)}

def pretokenize_input(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
   
   bytes_to_count = defaultdict(int)
   
   # Create a regex pattern to split on special tokens if provided
   special_token_pattern = None
   escaped_tokens = [re.escape(token) for token in special_tokens]
   if len(escaped_tokens) > 0:
      special_token_pattern = re.compile('|'.join(escaped_tokens))
    
   with open(input_path, 'r', encoding='utf-8') as f:
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

def merge_tokens(pretokenized_bytes_to_count: dict[tuple[bytes], int],
                 number_merges: int,
                 init_vocab: dict[int, bytes]) -> \
   tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
   # we want to merge the pretokenized tokens, updating our vocab
   # and returning a list of merges
   vocab = None
   merges = []
   return vocab, merges

def add_special_tokens(vocab, special_tokens):
   for i, token in enumerate(special_tokens):
      vocab[i + 256] = token.encode('utf-8')

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> \
   tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
      initial_vocab = initialize_vocabulary()
      pretokenized_bytes_to_count = pretokenize_input(input_path)
      
      number_merges = vocab_size - len(initial_vocab) - len(special_tokens)
      vocab, merges = merge_tokens(pretokenized_bytes_to_count, number_merges, vocab_size)
      add_special_tokens(vocab, special_tokens)
      
      return vocab, merges
      