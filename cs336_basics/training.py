import torch
from torch.amp import GradScaler, autocast
import os
import numpy as np
import argparse
from tqdm import tqdm
import yaml
import wandb
import time
import json

from cs336_basics.data_loader import get_batch
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, grad_clipping, learning_rate_schedule
from cs336_basics.loss import cross_entropy_loss, perplexity

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

def log_validation(model, val_data, args, iteration, start_time, val_log, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_perplexity = 0
        for _ in range(8):
            val_inputs, val_targets = get_batch(val_data, args.batch_size, args.context_length, device)
            val_outputs = model(val_inputs)
            val_loss += cross_entropy_loss(val_outputs, val_targets)
            val_perplexity += perplexity(val_outputs, val_targets).item()
        val_loss /= 8
        val_perplexity /= 8

    current_time = time.time() - start_time
    val_entry = {
        'iteration': iteration,
        'val_loss': val_loss.item(),
        'val_perplexity': val_perplexity,
        'elapsed_time': current_time,
    }
    val_log.append(val_entry)
    print(f"[Validation] Iteration {iteration}, Loss: {val_loss.item():.4f}, Perplexity: {val_perplexity:.2f}, Time: {current_time:.2f}s")
    if args.use_wandb:
        wandb.log(val_entry, step=iteration)

def main(args):
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, val in config.items():
            if key == "lr" or key == "weight_decay" or key == "rope_theta" or key == "eps":
                setattr(args, key, float(val))
            elif key == "batch_size" or key == "context_length" or key == "num_iterations" or key == "log_freq" or key == "val_freq" or key == "ckpt_freq" or key == "vocab_size" or key == "d_model" or key == "num_heads" or key == "d_ff" or key == "num_layers":
                setattr(args, key, int(val))
            elif key == "betas":
                setattr(args, key, tuple(float(x) for x in val))
            elif key == "no_rope" or key == "post_norm" or key == "no_norm" or key == "only_silu":
                setattr(args, key, bool(val))
            else:
                setattr(args, key, val)

    if args.use_wandb:
        wandb.init(project=args.project, name=args.experiment_name, config=vars(args))

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    train_data = np.memmap(args.train_data, mode='r', dtype=np.uint16)
    if args.val_data:
        val_data = np.memmap(args.val_data, mode='r', dtype=np.uint16)
    else:
        val_data = None

    token_positions = torch.arange(args.context_length)
    print(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.rope_theta)
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
        no_rope=args.no_rope,
        post_norm=args.post_norm,
        no_norm=args.no_norm,
        only_silu=args.only_silu
    )
    torch.compile(model)
    model.to(device)

    # build the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      eps=args.eps,
                      betas=args.betas)

    starting_iteration = 0
    if args.checkpoint:
        starting_iteration = load_checkpoint(args.checkpoint, model, optimizer)

    train_log = []
    val_log = []

    # function of the number of iterations
    args.ckpt_freq = args.num_iterations // 10
    args.val_freq = args.num_iterations // 1000
    args.log_freq = args.num_iterations // 1000

    # function of the number of iterations
    args.warmup_steps = args.num_iterations // 10

    scaler = GradScaler()

    # training loop
    print("Starting training loop...")
    start_time = time.time()
    for iteration in tqdm(range(starting_iteration, args.num_iterations + 1), desc="Training"):
        # Runs the forward pass with autocasting.
        model.train()
        optimizer.zero_grad()

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)  # outputs shape: (batch_size, context_length, d_model)
            loss = cross_entropy_loss(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_clipping(model.parameters(), args.max_grad_norm)
        new_lr = learning_rate_schedule(iteration, a_max=args.lr, a_min=0.1*args.lr, t_w=args.warmup_steps, t_c=args.num_iterations)
        for group in optimizer.param_groups:
            group['lr'] = new_lr

        scaler.step(optimizer)
        scaler.update()

        if iteration % args.log_freq == 0:
            train_perplexity = perplexity(outputs, targets).item()
            # for wandb logging
            current_time = time.time() - start_time
            log_entry = {
                'iteration': iteration,
                'train_loss': loss.item(),
                'train_perplexity': train_perplexity,
                'elapsed_time': current_time,
                'learning_rate': new_lr
            }
            train_log.append(log_entry)

            print(f"Iteration {iteration}/{args.num_iterations}, Training Loss: {loss.item():.4f}, Perplexity: {train_perplexity:.2f}, Time: {current_time:.2f}s")
            if args.use_wandb:
                wandb.log(log_entry, step=iteration)

        # evaluate on the validation dataset
        if val_data is not None and iteration % args.val_freq == 0:
            log_validation(model, val_data, args, iteration, start_time, val_log, device)

        # checkpointing
        if iteration % args.ckpt_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.experiment_name}_model_iter_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        if time.time() - start_time >= 5280:
            print("reached max time limit")
            log_validation(model, val_data, args, iteration, start_time, val_log, device)
            break

    # save the logs just in case
    log_file = os.path.join(args.ckpt_dir, f"{args.experiment_name}_log.json")
    with open(log_file, "w", encoding="utf-8") as lf:
        json.dump(train_log, lf, indent=2)
    print(f"Experiment log saved to {log_file}")

    # compute the validation loss across all data
    num_batches=(val_data.shape[0]-1) // (args.batch_size * args.context_length)
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        for i in range(num_batches):
            start = i * args.batch_size * args.context_length
            end = start + args.batch_size * args.context_length + 1
            batch = val_data[start:end].to(device).view(args.batch_size, args.context_length+1)
            loss = cross_entropy_loss(model(batch[:,:-1]), batch[:,1:])
            total_loss += loss.item() * args.batch_size * args.context_length
            total_tokens += args.batch_size * args.context_length
    avg_loss = total_loss/total_tokens
    print(f"Validation Loss: {avg_loss:.4f}")

    # finish and close wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Transformer model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
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

    # LR scheduling and grad clipping
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")

    # training args
    parser.add_argument("--context_length", type=int, default=128, help="Context length for each training example")
    parser.add_argument("--ckpt_dir", type=str, default="/home/c-aalag/results/checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (in iterations)")
    parser.add_argument("--val_freq", type=int, default=500, help="Validation frequency (in iterations)")
    parser.add_argument("--ckpt_freq", type=int, default=1000, help="Checkpoint saving frequency (in iterations)")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    # wandb args
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--project", type=str, default="transformer_lm_experiments", help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, default="baseline_experiment", help="Wandb experiment name")

    # checkpoint args
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")

    # data args
    parser.add_argument("--train_data", type=str, help="Path to training data (np.memmap file)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data (np.memmap file)")

    # ablation arg
    # no-rope, post-norm, no-norm, silu
    parser.add_argument('--no_rope', action='store_true', help='No Rope')
    parser.add_argument('--post_norm', action='store_true', help='Post Norm')
    parser.add_argument('--no_norm', action='store_true', help='No Norm')
    parser.add_argument('--only_silu', action='store_true', help='Only SiLU')

    args = parser.parse_args()

    # make the checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)