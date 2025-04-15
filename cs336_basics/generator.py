import torch
from .base_layers import softmax
import argparse
from .transformer import TransformerLM
from .bpe_tokenizer import BPETokenizer

def top_p_sample(probs, top_p):
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # remove tokens that exceed top_p, keep first token
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., :1] = False

    # zero out probabilities to remove and renormalize
    filtered = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    return filtered / filtered.sum(dim=-1, keepdim=True)

def generate(model, prompt_tokens, max_tokens, context_length, eos_token_idx, device, temperature=0.0, top_p=0.9):
    model.eval()

    with torch.no_grad():
        tokens = torch.tensor(prompt_tokens, device=device)
        for _ in range(max_tokens):
            # truncate context length, run model, get last token logits
            tokens = tokens[-context_length:]
            logits = model(tokens)
            last_token_logits = logits[..., -1, :]

            # temperatured softmax
            probs = softmax(last_token_logits / temperature, dim=-1)

            # top_p sampling
            probs = top_p_sample(probs, top_p)
            next_token_index = torch.multinomial(probs, num_samples=1).item()
            next_token = torch.tensor([next_token_index], device=device)
            tokens = torch.cat([tokens, next_token])

            # check end of sequence
            if next_token == eos_token_idx:
                break

    return tokens

def main():
    parser = argparse.ArgumentParser(description="Generate text from a model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--eos_token_idx", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # model architecture -- this is fixed
    vocab_size = 50257
    num_layers = 12
    d_model = 512
    num_heads = 8
    d_ff = 2048
    rope_theta = 1e6
    no_rope = post_norm = no_norm = only_silu = False

    token_positions = torch.arange(args.context_length)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        token_positions=token_positions,
        device=args.device,
        dtype=torch.float32,
        no_rope=no_rope,
        post_norm=post_norm,
        no_norm=no_norm,
        only_silu=only_silu,
    )
    model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(args.tokenizer_path, args.tokenizer_path + ".merges", special_tokens)

    if args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt)
    else:
        prompt_tokens = [tokenizer.token_to_id["<|endoftext|>"]]

    generated_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        eos_token_idx=args.eos_token_idx,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    generated_text = tokenizer.decode(generated_tokens)

    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()