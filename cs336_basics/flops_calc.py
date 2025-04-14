import argparse

def calc_ffn_flops(d_model, num_layers, seq_len):
    return 16 * seq_len * num_layers * d_model * d_model

def calc_attention_flops(d_model, num_layers, seq_len):
    return 8 * num_layers * seq_len * d_model * d_model + 4 * num_layers * seq_len * seq_len * d_model

def calc_output_embed_flops(d_model, seq_len, vocab_size):
    return 2 * seq_len * vocab_size * d_model

def calc_flops(d_model, num_layers, seq_len, vocab_size):
    ffn_flops = calc_ffn_flops(d_model, num_layers, seq_len)
    print(f"FFN FLOPS: {ffn_flops:,} ({ffn_flops/1e9:.2f} GFLOPS)" if ffn_flops < 1e12 else f"FFN FLOPS: {ffn_flops:,} ({ffn_flops/1e12:.2f} TFLOPS)")
    attention_flops = calc_attention_flops(d_model, num_layers, seq_len)
    print(f"ATTENTION FLOPS: {attention_flops:,} ({attention_flops/1e9:.2f} GFLOPS)" if attention_flops < 1e12 else f"ATTENTION FLOPS: {attention_flops:,} ({attention_flops/1e12:.2f} TFLOPS)")
    output_embed_flops = calc_output_embed_flops(d_model, seq_len, vocab_size)
    print(f"OUTPUT EMBED FLOPS: {output_embed_flops:,} ({output_embed_flops/1e9:.2f} GFLOPS)" if output_embed_flops < 1e12 else f"OUTPUT EMBED FLOPS: {output_embed_flops:,} ({output_embed_flops/1e12:.2f} TFLOPS)")
    total_flops = ffn_flops + attention_flops + output_embed_flops
    print(f"TOTAL FLOPS: {total_flops:,} ({total_flops/1e9:.2f} GFLOPS)" if total_flops < 1e12 else f"TOTAL FLOPS: {total_flops:,} ({total_flops/1e12:.2f} TFLOPS)")

    # print percentages
    print(f"FFN FLOPS: {ffn_flops / (ffn_flops + attention_flops + output_embed_flops) * 100}%")
    print(f"ATTENTION FLOPS: {attention_flops / (ffn_flops + attention_flops + output_embed_flops) * 100}%")
    print(f"OUTPUT EMBED FLOPS: {output_embed_flops / (ffn_flops + attention_flops + output_embed_flops) * 100}%")
    return ffn_flops + attention_flops + output_embed_flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    print(args)
    calc_flops(args.d_model, args.num_layers, args.seq_len, args.vocab_size)
