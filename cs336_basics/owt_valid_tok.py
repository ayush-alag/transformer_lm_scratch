from .bpe_trainer import BPETrainer, save_vocab_and_merges, report_memory_usage

# 32000, 12 hours max
owt_trainer = BPETrainer(32000, "/data/a1-basics/owt_valid.txt", ["<|endoftext|>"], 8)
vocab, merges = owt_trainer.train_bpe("/home/c-aalag/results/owt_valid_profile")
save_vocab_and_merges(vocab,
                      merges,
                      "/home/c-aalag/results/owt_valid_vocab.json",
                      "/home/c-aalag/results/owt_valid_merges.txt")
report_memory_usage()

# srun --partition=a1-batch --qos=a1-batch-qos --gpus=1 --time=2:00:00 --pty bash -c "uv run python -m cs336_basics.tiny_stories_valid_tok > /home/c-aalag/results/valid_stdout.log"