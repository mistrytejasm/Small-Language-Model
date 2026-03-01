"""
================================================================================
SLM Data Processing File (data.py)
================================================================================

[Sequence: 2 of 6]

What this file is about:
This script is responsible for downloading English stories from HuggingFace, 
translating the raw English words into mathematical numbers (tokens), and saving 
those numbers onto your hard drive so the model can train on them efficiently.

How this code works step by step:
1. We load the "TinyStories" dataset using the `datasets` library.
2. We initialize the `tiktoken` byte-pair encoding algorithm (the same one GPT-2 uses).
3. The `process` function takes a story and converts the words into a list of numbers (IDs).
4. The script checks if `train.bin` exists. If not, it tokenizes the whole dataset across 
   all CPU cores and chunks the numbers into a massive binary (.bin) file on your hard drive.
5. The `get_batch` function is used during training. It opens the binary file on the 
   hard drive, grabs a random chunk of numbers based on `block_size` and `batch_size`, 
   and ships them to the GPU for training. 
   X = Input words. Y = the expected "Next" words.
"""

import os
import numpy as np
import torch
from datasets import load_dataset
import tiktoken
from tqdm.auto import tqdm

from config import block_size, batch_size, device, device_type

# 1. Initialize Tokenizer
# gpt2 is a standard fast byte-pair encoding tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    """Encodes standard English text into numeric Token IDs."""
    ids = enc.encode_ordinary(example['text']) 
    out = {'ids': ids, 'len': len(ids)}
    return out

def prepare_data():
    """Downloads dataset and saves tokenized arrays to massive .bin files on disk."""
    print("\n" + "="*50)
    print("PREPARING DATA PIPELINE")
    print("="*50)

    # Ensure a dedicated data folder exists to keep the root directory clean
    os.makedirs("data", exist_ok=True)
    
    train_path = os.path.join("data", "train.bin")
    val_path = os.path.join("data", "validation.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("[SUCCESS] Data is already tokenized and stored in binary files. Skipping download.")
        return

    print("\n[Step 1] Downloading 'TinyStories' dataset from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories")

    print("\n[Step 2] Translating English words into mathematical Token IDs...")
    print("Starting Multi-Core processing (this might take a few minutes):")
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing database",
        num_proc=8, # Uses 8 parallel processes to speed up mapping
    )

    # 2. Iterate over train and validation sets and write them out
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16 # max token value is ~50256, which fits in 16-bit integers
        
        # Select correct path
        filepath = train_path if split == 'train' else val_path
        
        # We use memory mapped files so we don't load gigabytes of data into your 8GB RAM
        print(f"\n[Step 3] Creating memory-mapped file for {filename}...")
        arr = np.memmap(filepath, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename} to disk', unit="batch"):
            # Batch together samples for faster hard-drive writing
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            
            # Write chunk into the mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            
        arr.flush()
        print(f"[SUCCESS] Successfully written and saved {filename}")

    print("\n" + "="*50)
    print("DATA PROCESSING COMPLETE!")
    print("The training data is now ready. You can safely run `python train.py`.")
    print("="*50 + "\n")

def get_batch(split):
    """
    Grabs a random batch of data to feed the neural network.
    Because of limited system RAM, this dynamically reads direct from the hard drive (.bin) inside the 'data/' folder.
    """
    # Recreate the memmap to read the data
    filename = f"{split}.bin" if split == 'train' else "validation.bin"
    filepath = os.path.join("data", filename)
    data = np.memmap(filepath, dtype=np.uint16, mode='r')
        
    # Grab random starting positions in the data file
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # X is the context, Y is the expected next token (offset by 1)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin_memory allows much faster asynchronous transfer from CPU RAM -> GPU VRAM
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        
    return x, y

if __name__ == '__main__':
    # If a user runs `python data.py` directly, it will prepare the data
    prepare_data()
