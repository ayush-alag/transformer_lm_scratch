from .bpe_trainer import BPETrainer, save_vocab_and_merges, report_memory_usage

# 32000, 12 hours max
owt_trainer = BPETrainer(32000, "/data/a1-basics/owt_train.txt", ["<|endoftext|>"], 8)
vocab, merges = owt_trainer.train_bpe("/home/c-aalag/results/owt_train_profile")
save_vocab_and_merges(vocab,
                      merges,
                      "/home/c-aalag/results/owt_train_vocab.json",
                      "/home/c-aalag/results/owt_train_merges.txt")
report_memory_usage()