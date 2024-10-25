import torch

# training params
block_size = 256  # maximum context
batch_size = 64  # independent sequences in parallel 
max_iters = 5000
eval_interval = 300
eval_iters = 200
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# model params
dropout = 0.2
n_embed = 384
n_blocks = 6
n_head = 6

# data processing
def load_text(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    return text

def create_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    return chars, vocab_size, stoi, itos

def create_encoders(stoi, itos):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode