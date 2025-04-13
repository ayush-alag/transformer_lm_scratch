import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
from .data_loader import get_batch
from .transformer import TransformerLM
from .optimizer import AdamW
from .loss import cross_entropy
def save_checkpoint(model, optimizer, iteration, out) :
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(obj, out)

def load_checkpoint(src, model, optimizer) :
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']

def main(args):
    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    train_data = np.memmap(args.train_data, mode='r', dtype=np.int64)
    if args.val_data:
        val_data = np.memmap(args.val_data, mode='r', dtype=np.int64)
    else:
        val_data = None

    # TODO: is this correct for token_positions?
    token_positions = np.arange(args.context_length)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        token_positions=token_positions,
        device=device,
        dtype=torch.float32,  # TODO: typically your model parameters are float32
    )

    # build the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=args.betas)

    starting_iteration = 0
    if args.checkpoint:
        starting_iteration = load_checkpoint(args.checkpoint, model, optimizer)

    # training loop
    print("Starting training loop...")
    for iteration in tqdm(range(starting_iteration, args.num_iterations + 1), desc="Training"):
        model.train()
        optimizer.zero_grad()

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        outputs = model(inputs)  # outputs shape: (batch_size, context_length, d_model)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}/{args.num_iterations}, Training Loss: {loss.item():.4f}")

        # evaluate on the validation dataset
        if val_data is not None and iteration % args.val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = get_batch(val_data, args.batch_size, args.context_length, device)
                val_outputs = model(val_inputs)
                val_loss = cross_entropy(val_outputs, val_targets)
            print(f"[Validation] Iteration {iteration}, Loss: {val_loss.item():.4f}")

        # checkpointing
        if iteration % args.ckpt_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"model_iter_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Transformer model")
    # model args
    parser.add_argument("--d_model", type=int, default=512, help="Dimensionality of model features")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimensionality of the feed-forward network")
    parser.add_argument("--rope_theta", type=float, default=1e6, help="Rope theta")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")

    # optimization args
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="Betas for AdamW")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")

    # training args
    parser.add_argument("--context_length", type=int, default=128, help="Context length for each training example")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (in iterations)")
    parser.add_argument("--val_freq", type=int, default=500, help="Validation frequency (in iterations)")
    parser.add_argument("--ckpt_freq", type=int, default=1000, help="Checkpoint saving frequency (in iterations)")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    # checkpoint args
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")

    # data args
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (np.memmap file)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data (np.memmap file)")
    args = parser.parse_args()

    # make the checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)