import torch
from cs336_basics.base_layers import softmax
import argparse
from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe_tokenizer import BPETokenizer

def top_p_sample(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # remove tokens that exceed top_p, keep first token
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., :1] = False
    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # zero out probabilities to remove and renormalize
    sampled_sorted_index = torch.multinomial(sorted_probs, num_samples=1)
    next_token_index = sorted_indices.gather(dim=-1, index=sampled_sorted_index)
    return next_token_index.item()

def generate(model, prompt_tokens, max_tokens, context_length, eos_token_idx, device, tokenizer, temperature=0.0, top_p=0.9):
    model.eval()

    with torch.no_grad():
        string_builder = tokenizer.decode(prompt_tokens)
        for _ in range(max_tokens):
            tokens = torch.tensor([prompt_tokens], device=device)
            # truncate context length, run model, get last token logits
            tokens = tokens[-context_length:]
            logits = model(tokens)
            last_token_logits = logits[..., -1, :]

            # temperatured softmax
            probs = softmax(last_token_logits / temperature, dim=-1)

            # top_p sampling
            next_token_index = -1
            while next_token_index == -1 or next_token_index == 10:
                next_token_index = top_p_sample(probs, top_p)

            next_token = torch.tensor([next_token_index], device=device)
            prompt_tokens.append(next_token_index)
            string_builder += tokenizer.decode([next_token_index])

            # check end of sequence
            if next_token == eos_token_idx:
                break

    return string_builder, prompt_tokens

def main():
    parser = argparse.ArgumentParser(description="Generate text from a model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--eos_token_idx", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # model architecture -- this is fixed
    vocab_size = args.vocab_size
    num_layers = 4
    d_model = args.d_model
    num_heads = 16
    d_ff = args.d_ff
    rope_theta = 10000
    no_rope = post_norm = no_norm = only_silu = False

    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path, special_tokens)

    if args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt)
    else:
        prompt_tokens = [tokenizer.bytes_to_ids["<|endoftext|>"]]

    token_positions = torch.arange(args.context_length, device=args.device)
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
        no_rope=False,
        post_norm=post_norm,
        no_norm=no_norm,
        only_silu=only_silu,
    )
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model"])

    # model = torch.compile(model)
    model.to(args.device)

    generated_text, prompt_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        eos_token_idx=args.eos_token_idx,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        tokenizer=tokenizer,
    )

    print("Generated text:")
    print(generated_text)

    # print("Prompt tokens:")
    # print(prompt_tokens)

if __name__ == "__main__":
    main()

# srun --partition=interactive --qos=interactive-qos --gpus=1 --pty bash -c "uv run cs336_basics/generator.py --checkpoint /home/c-aalag/results/checkpoints/nbatch_32_model_iter_40000.pt --vocab_path /home/c-aalag/results/owt_train_vocab.json --merges_path /home/c-aalag/results/owt_train_merges.txt --max_tokens 256 --prompt 'The cat said'"
# srun --partition=interactive --qos=interactive-qos --gpus=1 --pty bash -c "uv run cs336_basics/generator.py --checkpoint /home/c-aalag/results/checkpoints/nbatch_32_model_iter_40000.pt --vocab_path /home/c-aalag/results/tiny_train_vocab.json --merges_path /home/c-aalag/results/tiny_train_merges.txt --max_tokens 256 --prompt 'The cat said'"

#srun --partition=interactive --qos=interactive-qos --gpus=1 --pty bash -c "uv run cs336_basics/generator.py --checkpoint /home/c-aalag/results/checkpoints/owt_base_model_iter_20000.pt --vocab_path /home/c-aalag/results/owt_train_vocab.json --merges_path /home/c-aalag/results/owt_train_merges.txt --max_tokens 256 --prompt 'The cat said' --vocab_size 32000 --d_model 512 --d_ff 1344"

# srun --partition=interactive --qos=interactive-qos --gpus=1 --pty bash -c "uv run cs336_basics/generator.py --checkpoint /home/c-aalag/results/checkpoints/owt_1024_1e3_model_iter_6000.pt --vocab_path /home/c-aalag/results/owt_train_vocab.json --merges_path /home/c-aalag/results/owt_train_merges.txt --max_tokens 256 --prompt 'The cat said' --vocab_size 32000 --d_model 1024 --d_ff 2752"