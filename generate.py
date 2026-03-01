"""
================================================================================
SLM Generation Script (generate.py)
================================================================================

[Sequence: 6 of 6]

What this file is about:
This script allows you to talk to your AI. Once the model is trained 
and weights are saved in the `models/` folder, this script loads those weights 
into your computer's memory, takes a sentence you type in, and tries to predict 
how that story should finish.

How this code works step by step:
1. It imports the architectural layout of the model from `config.py` and `model.py`.
2. It looks into `models/best_model_params.pt` and loads the learned "weights" 
   (the math that makes it smart) into the empty neural network structure.
3. It takes your prompt (e.g., "A little girl went to the woods").
4. Uses `tiktoken` to convert your English words into numeric Token IDs.
5. Shoves those numbers into the model and asks it to `generate` up to 200 more words.
6. The `temperature` setting controls creativity. 
   - 0.1 = Very boring, repetitive, but highly grammatical.
   - 1.0 = Highly creative, but might go off-topic.
7. Decodes the numbers back into English and prints the story to your screen!
"""

import os
import torch
import tiktoken
from config import GPTConfig, device
from model import GPT

BEST_MODEL_PATH = "models/best_model_params.pt"

def load_trained_model():
    """Builds the network and slots the saved brain power (weights) into it."""
    
    if not os.path.exists(BEST_MODEL_PATH):
        print("Error: Could not find " + BEST_MODEL_PATH)
        print("You must run `python 04_train.py` and let it train before you can generate text!")
        return None

    print("Loading SLM configurations and spinning up neural network...")
    config = GPTConfig()
    model = GPT(config)
    
    print("Loading trained brain weights from " + BEST_MODEL_PATH + "...")
    # Map location ensures that even if you trained on GPU, you can infer on CPU if needed
    state_dict = torch.load(BEST_MODEL_PATH, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state_dict)
    
    # Put the model in evaluation mode (turns off dropout and training features)
    model.eval()
    model.to(device)
    
    return model

def generate_story(model, prompt, max_words=200, temperature=1.0, top_k=50):
    """
    Translates your prompt into numbers, feeds it to the model, and translates 
    the model's numeric guesses back into English.
    """
    print("\n[Prompt]: " + prompt)
    print("-" * 50)
    
    # 1. English -> Tokens
    enc = tiktoken.get_encoding("gpt2")
    
    # The unsqueeze command adds a "Batch" dimension because the model always expects batches of data
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(dim=0).to(device)
    
    # 2. Let the Neural Network do its magic
    # No gradient means PyTorch won't try to store memory for training
    with torch.no_grad():
        generated_ids = model.generate(context, max_new_tokens=max_words, temperature=temperature, top_k=top_k)
        
    # 3. Tokens -> English
    # The squeeze command removes the "Batch" dimension so we just have a list of words
    generated_text = enc.decode(generated_ids.squeeze().tolist())
    
    print("[AI Story]:\n" + generated_text)
    print("-" * 50)


if __name__ == '__main__':
    # Initialize the model once
    trained_model = load_trained_model()
    
    if trained_model is not None:
        # Loop to let the user keep typing prompts!
        print("\nWelcome to your custom Small Language Model (SLM)!")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("Start a story with a few words: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            generate_story(
                model=trained_model, 
                prompt=user_input, 
                max_words=150,    # How long the story should be
                temperature=0.8,  # A good balance of creative and coherent
                top_k=40          # Prevent the model from picking completely random garbage words
            )
