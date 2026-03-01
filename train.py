"""
================================================================================
SLM Training Script (train.py)
================================================================================

[Sequence: 4 of 6]

What this file is about:
This script brings the Small Language Model to life. It loads your neural network 
(`model.py`), feeds it the dataset (`data.py`), checks its mathematical guesses against 
reality, and slowly updates its weights so it gets smarter over time.

How this code works step by step:
1. `estimate_loss`: A helper function that pauses training and tests the model on both 
   the training data and unseen validation data. This proves the model is actually 
   learning English and not just memorizing the test answers.
2. We initialize the Model, Optimizer (AdamW), and a Learning Rate Scheduler. 
   - A Learning Rate Scheduler creates a "warm up" period where the model learns 
     slowly at first so it doesn't break, then speeds up, then slows down again.
3. Optimization tricks inside the Training Loop:
   a. Gradient Accumulation: Because you only have 4GB VRAM (GTX 1650), we can 
      only look at 8 sentences at a time. The script does the math for 16 mini-batches, 
      adds them all up, and then updates the weights once. This simulates a massive 
      GPU with 128 batch size!
   b. Mixed Precision (`torch.amp.autocast`): Does half the calculations in 16-bit 
      floating point instead of 32-bit. This uses half the VRAM and runs faster.
   c. Adaptive Learning Rate (ReduceLROnPlateau): If the model stops learning for 
      several evaluations, we automatically drop the learning rate to help it settle 
      into the optimal weights.
   d. Early Stopping (Patience): If the model completely stops improving on unseen 
      data (validation loss flatlines or rises), training will automatically abort 
      to save your time, keeping only the best model.
4. Saving Progress: Every 500 steps, we evaluate the loss. If it's the lowest loss 
   we've seen so far, we save the model to the `models/` folder.
5. Visualization Logs: All data (loss, accuracy, learning rate) is stored in a 
   `logs/` folder so we can graph it later and save the visualization images.
"""

import os
import json
import torch
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *
from data import get_batch
from model import GPT

# Ensure models and logs directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
BEST_MODEL_PATH = "models/best_model_params.pt"
LOGS_PATH = "logs/training_logs.json"

# Early Stopping Configuration
patience = 10  # How many evaluation steps to wait before stopping if no improvement

# Set up Mixed Precision context based on config.py
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

@torch.no_grad()
def estimate_loss(model):
    """
    Temporarily pauses training (`model.eval()`) to accurately gauge how 
    well the model is learning on 500 random batches without calculating gradients.
    """
    out = {'train_loss': 0, 'val_loss': 0, 'train_acc': 0, 'val_acc': 0}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split) # Fetch standard batch from the hard drive
            with ctx:
                _, loss, acc = model(X, Y)
            losses[k] = loss.item()
            accuracies[k] = acc.item()
            
        out[f'{split}_loss'] = losses.mean().item()
        out[f'{split}_acc'] = accuracies.mean().item()
        
    model.train() # Turn training features back on (like Dropout layers)
    return out

def train():
    # --- GPU Safety Check ---
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] GPU (CUDA) is not available! Training aborted to prevent extremely slow CPU training.")
    
    # 1. Boot up the network structure
    config = GPTConfig()
    
    # --- Print Configuration ---
    print("\n" + "="*50)
    print("SLM Training Configuration")
    print("="*50)
    print(f"Device Target:        {torch.cuda.get_device_name(0)}")
    print(f"Precision:            {dtype} (Mixed Precision)")
    print(f"Params (Millions):    ~{(config.vocab_size * config.n_embd + config.n_layer * (12 * config.n_embd**2 + 13 * config.n_embd) + config.vocab_size * config.n_embd) / 1e6:.1f}M")
    print("-" * 50)
    print(f"Block Size (Context): {config.block_size}")
    print(f"Batch Size:           {batch_size}")
    print(f"Grad Accumulation:    {gradient_accumulation_steps}")
    print(f"Effective Batch Size: {batch_size * gradient_accumulation_steps}")
    print("-" * 50)
    print(f"Learning Rate:        {learning_rate} (Max) -> {min_lr} (Min)")
    print(f"Max Training Iter:    {max_iters}")
    print(f"Warmup Steps:         {warmup_steps}")
    print("="*50 + "\n")

    print(f"Initializing GPT on device: {device}")
    model = GPT(config)
    model.to(device)

    # 2. Setup Optimizer (AdamW) with weight decay
    # Weight decay adds a small penalty to large weights, keeping the network grounded
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)

    # 3. Setup Learning Rate Scheduler (Adaptive)
    # ReduceLROnPlateau will monitor our validation loss. If the loss stops dropping
    # for `patience` rounds, it naturally cuts the learning rate by a factor of 0.5.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=min_lr)

    # 4. GradScaler
    # Because we use 16-bit numbers, tiny numbers can hit absolute zero. The scaler artificially inflates 
    # the numbers specifically during backpropagation to keep the math working.
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    # Tracking Variables
    best_val_loss = float('inf')
    patience_counter = 0
    logs = {"steps": [], "train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    print("\n" + "-"*50)
    print(f"Starting Training Matrix! Looking to process {max_iters} iterations.")
    print("Press Ctrl+C at any time to abort. (The best model is auto-saved).")
    print("-"*50 + "\n")
    
    # Setting up the live progress bar
    pbar = tqdm(range(max_iters), desc="Training Progress", unit="step", dynamic_ncols=True)
    
    for epoch in pbar:
        
        # [EVALUATION & SAVING PHASE]
        if epoch % eval_iters == 0 and epoch != 0:
            pbar.set_description(f"Evaluating Model Accuracy...")
            metrics = estimate_loss(model)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print a clear block to interrupt the progress bar and show the specific stats
            tqdm.write("\n" + "-"*50)
            tqdm.write(f"[Step {epoch}] Evaluation Results:")
            tqdm.write(f"  > Train Loss:     {metrics['train_loss']:.4f}")
            tqdm.write(f"  > Val Loss:       {metrics['val_loss']:.4f}")
            tqdm.write(f"  > Val Accuracy:   {metrics['val_acc']*100:.2f}%")
            tqdm.write(f"  > Learning Rate:  {current_lr:.6f}")
            tqdm.write("-" * 50)
            
            # Return progress bar description back to normal
            pbar.set_description("Training Progress")
            
            # Record data for visualize.py
            logs["steps"].append(epoch)
            logs["train_loss"].append(metrics['train_loss'])
            logs["val_loss"].append(metrics['val_loss'])
            logs["val_acc"].append(metrics['val_acc'])
            logs["lr"].append(current_lr)
            
            with open(LOGS_PATH, "w") as f:
                json.dump(logs, f)

            # Move the learning rate scheduler forward based on Validation Loss
            scheduler.step(metrics['val_loss'])

            # Did the model get smarter? Let's save it.
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                tqdm.write(f"[SAVE] New Best Model Saved (Loss: {best_val_loss:.4f}) -> {BEST_MODEL_PATH}")
                patience_counter = 0 # Reset patience
            else:
                patience_counter += 1
                tqdm.write(f"[WARNING] No improvement in validation loss for {patience_counter} evaluation(s).")
                
            # Early Stopping Triggered!
            if patience_counter >= patience:
                tqdm.write(f"\n[EARLY STOPPING] Validation loss hasn't improved in {patience} evals.")
                tqdm.write(f"We are stopping early at step {epoch} to save your time. The best model is safely saved!")
                break

        # [TRAINING PHASE]
        # Fetch the words
        X, y = get_batch("train")
        X, y = X.to(device), y.to(device)

        # Run the math
        with ctx:
            logits, loss, acc = model(X, y)
            # We scale the loss down, so when we add 16 of them together, it equals 1 whole batch
            loss = loss / gradient_accumulation_steps
            
        # Backpropagation (finding out how wrong the model was)
        scaler.scale(loss).backward()

        # [WEIGHT UPDATE PHASE]
        # Have we accumulated enough mini-batches? If so, perform the massive update.
        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            # Clip gradient explosions (stops weights from going to infinity)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Apply the updates to the neural network
            scaler.step(optimizer)
            scaler.update()
            
            # Erase the memory of the past gradients so we can start fresh
            optimizer.zero_grad(set_to_none=True)
            
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("Your best model weights are safely saved in the 'models/' folder.")
    print("You can now run `python visualize.py` to see the graphs, or `python generate.py` to talk to it!")
    print("="*50 + "\n")

if __name__ == '__main__':
    train()
