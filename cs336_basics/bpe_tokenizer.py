class BPETokenizer:
   def __init__(self, vocab, merges, special_tokens=None):
      self.vocab = vocab
      self.merges = merges
      self.special_tokens = special_tokens
   
   def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
      pass
   
   def encode(self, text):
      # Encode a list into a sequence of token IDs
      pass
   
   def encode(self, iterable):
      # iterable of strings (python file handle) -> generator that lazily yields token IDs
      pass
   
   def decode(self, ids):
      # Decode a list of token IDs into text
      pass