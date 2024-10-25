import torch
import os
from config import (
    block_size, batch_size, max_iters, eval_interval, 
    eval_iters, lr, device, n_head, n_blocks,
    load_text, create_vocab, create_encoders
)
from model import BigramModel

def get_batch(train_data, test_data, split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, test_data):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, test_data, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    # Set random seed
    torch.manual_seed(1337)
    torch.cuda.empty_cache()

    # Load and process data
    text = load_text("GPT_from_scratch/data/input.txt")
    chars, vocab_size, stoi, itos = create_vocab(text)
    encode, decode = create_encoders(stoi, itos)
    
    # Create dataset
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    test_data = data[n:]

    # Initialize model
    model = BigramModel(vocab_size=vocab_size, n_head=n_head, n_blocks=n_blocks)
    model = model.to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for steps in range(max_iters):
        if steps % eval_interval == 0:
            losses = estimate_loss(model, train_data, test_data)
            print(f"step {steps}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")
        
        xb, yb = get_batch(train_data, test_data, "train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate sample text
    context = torch.zeros((1,1), dtype=torch.long).to(device)
    generated_text = decode(model.generate(idx=context, max_new_tokens=500)[0].tolist())
    print("\nGenerated text:\n", generated_text)

if __name__ == "__main__":
    train()