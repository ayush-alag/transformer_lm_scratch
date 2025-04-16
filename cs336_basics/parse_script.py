#!/usr/bin/env python3
import json
import argparse

def convert_vocab(json_file: str) -> list[str]:
    """
    Load a JSON vocabulary file where each key maps to a list of ints,
    convert each list of ints to a UTF-8 string, and return a list of strings
    sorted by length (shortest first).

    Parameters:
      json_file: path to the vocabulary JSON file.

    Returns:
      A list of token strings sorted by length.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)

    tokens = []
    for key in sorted(vocab_dict.keys(), key=lambda k: int(k)):
        token_ints = vocab_dict[key]
        token_bytes = bytes(token_ints)
        try:
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            token_str = token_bytes.decode('utf-8', errors='replace')
        tokens.append(token_str)

    sorted_tokens = sorted(tokens, key=len, reverse=True)
    return sorted_tokens

def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSON vocabulary into a list of strings sorted by length."
    )
    parser.add_argument("json_file", type=str, help="Path to the JSON vocabulary file")
    parser.add_argument("output_file", type=str, help="Path for saving the sorted tokens")
    args = parser.parse_args()

    sorted_tokens = convert_vocab(args.json_file)

    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        for token in sorted_tokens:
            out_file.write(token + "\n")

    print(f"Converted {len(sorted_tokens)} tokens. Sorted vocabulary saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()