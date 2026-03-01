"""
================================================================================
SLM Project Config File (config.py)
================================================================================

[Sequence: 1 of 6]

What this file is about:
This file serves as the singular "control panel" for the entire Small Language 
Model (SLM). Instead of hunting through different files to change learning rates 
or model sizes, you only need to change them here.

How this code works step by step:
1. We import `dataclass` to easily structure our model parameters.
2. We define the `GPTConfig` class which holds the architectural layout 
   of the neural network (like how many layers and attention heads).
3. We define the training hyper-parameters, which control how the model learns 
   (like batch size, learning rate, and memory-saving techniques).
4. Because you are using an NVIDIA GTX 1650 with 4GB VRAM, the variables in this
   file are heavily optimized:
   - `block_size` is kept small (128 words/tokens).
   - `batch_size` is kept tiny (8 sequences at a time) to prevent Out of Memory.
   - `gradient_accumulation_steps` is set high (16) so the model still *learns*
     as if you had a massive GPU (batch size 8 * 16 = 128 effective batch size).
"""

from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    block_size: int = 128      # Context window limit (how many words it can look back at)
    vocab_size: int = 50257    # The size of our dictionary (from GPT-2 tokenizer)
    n_layer: int = 6           # Number of Transformer blocks on top of each other
    n_head: int = 6            # Number of attention "viewpoints" in each block
    n_embd: int = 384          # The size of the mathematical vector used to represent a word
    dropout: float = 0.1       # Randomly turns off 10% of neurons to prevent overfitting
    bias: bool = True          # True: use bias in neural network lines. False: slightly faster

# -----------------------------------------------------------------------------
# Training Hyper-parameters
# -----------------------------------------------------------------------------

# VRAM Optimization for GTX 1650 4GB
batch_size = 8                 # MUST BE SMALL (e.g. 4 or 8) to fit in 4GB VRAM
block_size = 128               # Context size, smaller saves memory.
gradient_accumulation_steps = 16 # Combines 16 small batches into 1 large update

# Learning Speeds
learning_rate = 1e-4           # How fast the model updates weights (1e-4 is safe/stable)
min_lr = 5e-4                  # The lowest the learning rate drops to
max_iters = 100000             # Total number of training steps (increased for longer training)
warmup_steps = 1000            # Gradually increases learning rate at the start
eval_iters = 500               # How often we evaluate performance and save

# Hardware Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Mixed Precision - This makes math slightly less exact (16-bit instead of 32-bit) 
# but cuts VRAM usage in half and runs twice as fast.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
