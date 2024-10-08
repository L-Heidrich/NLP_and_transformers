import torch.nn as nn 
from torch.nn import functional as F 
import torch
torch.manual_seed(1337)

torch.cuda.empty_cache() 

block_size= 8
batch_size=2
max_iters = 3000
eval_interval = 300
eval_iters = 200
lr = 1e-2
device = "cpu" if torch.cuda.is_available() else "cpu"
n_embed = 32

text = open("C:/Users/lenna/Desktop/Transformers/NLP_and_transformers/Bigram model/names.txt", "r").read().splitlines()

characters = sorted(list(set(text)))
vocab_size = len(characters)
vocab_size, "".join(characters)

stoi = {ch:i for i,ch in enumerate(characters)}
itos = {i:ch for i,ch in enumerate(characters)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
train_data[:block_size+1]

def get_batch(split):
    data= train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #Random sampling batch_size indexes from the dataset to use as starting points for chunks
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device) # Stacking the results up in a batch_size x chunk_size tensor
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


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # embedding for each position

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # Dimensions: Batch, Time, vocab size
        positional_embedding = self.position_embedding_table(torch.arange(T, device=device)) # integers until T-1, Embeeded to create a T, C matrix
        
        x = token_embeddings + positional_embedding
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
            logits, loss = self(idx)

            # Only the last time step will be considered 
            logits = logits[:, -1, :]

            # Get probablities
            probs = F.softmax(logits, dim=-1)

            # sampling from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            #appeding the results
            idx = torch.cat((idx, idx_next), dim=1)

        return idx    
    

model = BigramModel(vocab_size=vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

for steps in range(max_iters):

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']} train loss {losses['val']}")
    
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
context = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx=context, max_new_tokens=100)[0].tolist()))