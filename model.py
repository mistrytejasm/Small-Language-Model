"""
================================================================================
SLM Neural Network Architecture File (model.py)
================================================================================

[Sequence: 3 of 6]

What this file is about:
This is the mathematical heart of your Small Language Model (SLM). It defines 
the "Transformer" architecture—the exact same technology that powers ChatGPT, 
Llama, and Claude, just scaled down to fit your GTX 1650.

How this code works step by step:
1. `LayerNorm`: Normalizes data passing through layers so the network doesn't 
   explode with massive numbers or vanish to zero during training.
2. `CausalSelfAttention`: The "Attention" mechanism. This is how the AI learns context.
   It looks back at previous words to figure out what word should come next.
   - We check if your PyTorch version supports "Flash Attention" 
     (`scaled_dot_product_attention`). Flash Attention is a massively optimized 
     algorithm that runs up to 3x faster and uses way less VRAM.
3. `MLP`: A basic Feed-Forward neural network block that adds deep learning non-linearity
   after the attention step. Uses the GELU activation function.
4. `Block`: Stitches one Attention layer and one MLP layer together. Your config
   file asks for 6 of these blocks stacked on top of each other.
5. `GPT`: This is the final assembled engine. 
   - It takes in word tokens.
   - Projects them into mathematical embeddings (`wte` and `wpe`).
   - Runs them through the 6 Transformer blocks (`h`).
   - Calculates a vocabulary "loss" using Cross-Entropy if we are training.
   - Pushes out predicted next-word IDs using Multinomial probabilities if generating.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PyTorch Modules (The Building Blocks)
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Custom Layer Normalization with optional bias"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """The mechanism that allows words to look at previous words for context."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Calculate Query, Key, and Value all at once in one matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Optimization: Flash Attention uses significantly less VRAM (Crucial for 4GB GTX 1650)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # Fallback for older PyTorch versions
            print("WARNING: Flash Attention not found. Training will use more VRAM.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Context Length), Channels (Embedding Size)
        
        # Calculate Query (q), Key (k), and Value (v)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Extremely optimized C++ Attention kernel
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                               dropout_p=self.attn_dropout.p if self.training else 0.0, 
                                               is_causal=True)
        else:
            # Slower, manual attention calculation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Recombine heads back together
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """A standard feed-forward neural network attached to every Attention block."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    """One single Transformer block (Attention -> Norm -> MLP -> Norm)."""
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        # We process Attention first
        x = x + self.attn(self.ln1(x))
        # Then the Feed Forward network
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------------------------------------------------------------------
# The Main GPT Model
# -----------------------------------------------------------------------------

class GPT(nn.Module):
    """The master class that stitches the whole Small Language Model together."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Define all the parts of the model inside a dictionary
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd), # Word Token Embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd), # Positional Embeddings (Word order)
            drop=nn.Dropout(config.dropout),
            
            # This creates a list of 6 stacked Blocks based on config.n_layer
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        
        # The final layer that pushes sizes up to the 50,257 vocab limit
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Optimization: Weight Tying. Uses the exact same memory for Input and Output word vectors.
        # This saves millions of parameters and a massive amount of VRAM.
        self.transformer.wte.weight = self.lm_head.weight 

        # Initialize network weights properly instead of random noise
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """Mathematical formula to initialize weights around 0.0 with a tiny standard deviation."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """The core Loop that processes new words."""
        device = idx.device
        b, t = idx.size()
        
        # Make sure the input sentence isn't too long for our model
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 1. Look up the meaning/math of the words
        tok_emb = self.transformer.wte(idx) 
        
        # 2. Look up the order the words were written in
        pos_emb = self.transformer.wpe(pos) 
        
        # Combine them
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass the data through the 6 Transformer Blocks!
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # We are TRAINING
            # Calculate what we got vs what the real target word was (Loss)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Optional: Calculate basic accuracy (how often did we predict the EXACT right word)
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == targets).float().mean()
            
            return logits, loss, accuracy
        else:
            # We are INFERENCING (just chatting)
            # Only care about the very last word generated
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Takes a starting prompt and continuously loops the network to generate a full story.
        idx: Tensor of shape (B, T) - The starting prompt IDs
        """
        for _ in range(max_new_tokens):
            # Crop the prompt if it exceeds our memory limit (block_size)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Run the model forward
            logits, _ = self(idx_cond)
            
            # Predict the next word probabilities
            logits = logits[:, -1, :] / temperature
            
            # Top-k: Optionally cut off the really weird/rare word choices
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            # Pluck the final word out of the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Stick it onto our sentence and loop again!
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
