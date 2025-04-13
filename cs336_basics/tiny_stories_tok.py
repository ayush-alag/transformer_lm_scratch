from .bpe_trainer import BPETrainer, save_vocab_and_merges, report_memory_usage

tiny_stories_trainer = BPETrainer(10000, "/data/a1-basics/TinyStoriesV2-GPT4-train.txt", ["<|endoftext|>"], 8)
vocab, merges = tiny_stories_trainer.train_bpe("/home/c-aalag/results/tiny_train_trainer_profile")
save_vocab_and_merges(vocab,
                      merges,
                      "/home/c-aalag/results/tiny_train_vocab.json",
                      "/home/c-aalag/results/tiny_train_merges.txt")
report_memory_usage()