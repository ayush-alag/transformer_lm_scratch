from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.bpe_trainer import BPETrainer
import argparse
import json

def get_vocab_and_merges(vocab_file, merges_file):
    with open(vocab_file, 'r', encoding='utf-8') as vf:
        vocab = json.load(vf)
        vocab = {int(k): v for k, v in vocab.items()}

    merges = []
    with open(merges_file, 'r', encoding='utf-8') as mf:
        for line in mf:
            b1_str, b2_str = line.strip().split('\t')
            b1 = tuple(bytes([b]) for b in b1_str.split())
            b2 = tuple(bytes([b]) for b in b2_str.split())
            merges.append((b1, b2))
    return vocab, merges

def encode_text(input_file, output_file, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    tokenizer.serialize(token_ids, output_file)
        
    original_size = len(text.encode('utf-8'))
    # each token id is represented as uint16
    encoded_size = len(token_ids) * 2  
    compression_ratio = original_size / encoded_size
    
    print("Original size (bytes):", original_size)
    print("Encoded size (bytes):", encoded_size)
    print("Compression ratio: {:.2f}".format(compression_ratio))

def main():
    parser = argparse.ArgumentParser(description='Encode text using BPE tokenizer')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    parser.add_argument('--merges', required=True, help='Path to merges file')
    parser.add_argument('--input', required=True, help='Input text file to encode')
    parser.add_argument('--output', required=True, help='Output file for encoded text')
    args = parser.parse_args()

    vocab, merges = get_vocab_and_merges(args.vocab, args.merges)
    tokenizer = BPETokenizer(vocab, merges)
    encode_text(args.input, args.output, tokenizer)

if __name__ == '__main__':
    main()