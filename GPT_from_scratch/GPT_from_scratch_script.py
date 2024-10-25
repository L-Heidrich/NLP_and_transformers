import torch.nn as nn 
from torch.nn import functional as F 
import torch
import os

torch.manual_seed(1337)

torch.cuda.empty_cache() 

# training params
block_size= 256 # maximum context
batch_size= 64 # independent sequences in parallel 
max_iters = 5000
eval_interval = 300
eval_iters = 200
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

#model params
dropout = 0.2
n_embed = 384
n_blocks = 6
n_head = 6


with open("GPT_from_scratch/data/input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(chars, vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
train_data[:block_size+1]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _ , loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        
        B, T, C = x.shape

        k = self.key(x) # projection into B, T, head_size
        q = self.query(x) # projection into  B, T, head_size
        v = self.value(x)

        self_attention = q @ k.transpose(-2, -1) * C**-0.5 # affinities and scaling

        self_attention = self_attention.masked_fill(self.tril[:T, :T]==0, float('-inf')) # applying the mask
        self_attention = F.softmax(self_attention, dim=-1) # Softmax is a normalization operation, thats why the result is the same. 
        self_attention = self.dropout(self_attention)

        self_attention = self_attention @ v

        return self_attention

class MultiheadAttention(nn.Module):
    def __init__(self,  num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        multi_head_attention = torch.cat([h(x) for h in self.heads], dim=-1) # cat over channel dim
        multi_head_attention = self.proj(multi_head_attention) # projection back into the residual pathway
        multi_head_attention = self.dropout(multi_head_attention)

        return multi_head_attention

class FeedForward(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, n_embed*4), #projection up to *4 according to the paper
                                nn.ReLU(),
                                nn.Linear(n_embed*4, n_embed), # projection back into the residual pathway
                                nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultiheadAttention(n_head, head_size)
        self.ff = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x)) # Skip connection , Layernorm applied before self attention
        x = x + self.ff(self.ln2(x)) # Skip connection 
        return x

class BigramModel(nn.Module):
    def __init__(self, vocab_size, n_head, n_blocks):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # embedding for each position

        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head=n_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # Dimensions: Batch, Time, vocab size
        positional_embedding = self.position_embedding_table(torch.arange(T, device=device)) # integers until T-1, Embeeded to create a T, C matrix
        
        x = token_embeddings + positional_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss=None
        else:
            # Torch expects a different dimension for crossetropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            
            # last #block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)

            # Only the last time step will be considered 
            logits = logits[:, -1, :]

            # Get probablities
            probs = F.softmax(logits, dim=-1)

            # sampling from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #print(decode([idx_next[0].item()]), end='\n')
            #appeding the results
            idx = torch.cat((idx, idx_next), dim=1)

        return idx    
    

model = BigramModel(vocab_size=vocab_size, n_head=n_head, n_blocks=n_blocks)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

if True:
    for steps in range(max_iters):

        if steps % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {steps}: train loss {losses['train']} val loss {losses['val']}")
        
        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    
context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist()))