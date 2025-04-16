from cs336_basics.bpe_tokenizer import BPETokenizer
import argparse
import cProfile
import pstats
import io
import numpy as np

def encode_text(input_file, output_file, tokenizer, max_files=None):
    mm = np.memmap(input_file, mode='r', dtype='uint8')
    text = mm.tobytes().decode('utf-8', errors='replace')

    segments = text.split('<|endoftext|>')[:max_files]
    text_to_encode = '<|endoftext|>'.join(segments)

    token_ids = tokenizer.encode(text, max_chunks=max_files)
    tokenizer.serialize(token_ids, output_file)

def main():
    parser = argparse.ArgumentParser(description='Encode text using BPE tokenizer')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    parser.add_argument('--merges', required=True, help='Path to merges file')
    parser.add_argument('--input', required=True, help='Input text file to encode')
    parser.add_argument('--output', required=True, help='Output file for encoded text')
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, special_tokens)
    pr = cProfile.Profile()
    pr.enable()
    encode_text(args.input, args.output, tokenizer)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    print(s.getvalue())

if __name__ == '__main__':
    main()
