import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        #init param
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *=( 2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558
        }[model_type]
        config_args['vocab_size'] = 50257 #
        config_args['block_size'] = 1024 #
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]


        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
#

# Define device handling properly with fallback
device = "cpu"  # Default fallback
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    try:
        # Test MPS with a small tensor operation
        test_tensor = torch.ones(1).to("mps")
        test_tensor = test_tensor + 1
        device = "mps"
        print(f"MPS is available and working, using device: {device}")
    except Exception as e:
        print(f"MPS is available but not working properly: {e}")
        print("Falling back to CPU")
        device = "cpu"
else:
    print("Using CPU")

print(f"Using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#


num_return_sequences = 5
max_length = 30

#get data
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()

text = text[:1000]



tokens = enc.encode(text)

# Create batches
B, T = 4, 32  # batch size and sequence length
# Make sure we have enough tokens for a complete batch
if len(tokens) < B * T + 1:
    raise ValueError(f"Not enough tokens. Need at least {B*T+1}, but got {len(tokens)}")


#Chanege to buff here
# Create input-target pairs
x = torch.tensor(tokens[:(B*T)], dtype=torch.long).view(B, T)
y = torch.tensor(tokens[1:(B*T+1)], dtype=torch.long).view(B, T)


# Move to device AFTER creating tensors
x = x.to(device)
y = y.to(device)

# Initialize the model on the same device
model = GPT(GPTConfig())
model = model.to(device)
#oprimize
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)  # Pass y as targets
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Step {i}, Loss: {loss.item()}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o')
plt.title("Training Loss over Steps")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
    # Forward pass
try:
    logits, loss = model(x, targets=y)  # Pass y as targets
    print("logits shape:", logits.shape)
    print("loss:", loss.item() if loss is not None else None)
except RuntimeError as e:
    print(f"Error with MPS device: {e}")
    print("Falling back to CPU")
    device = "cpu"
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)
    logits, loss = model(x, targets=y)
    print("logits shape:", logits.shape)
    print("loss:", loss.item() if loss is not None else None)

#get logits
# model = GPT(GPTConfig())
# model.to(device)
# logits, loss= model(x)
# print("logits shape:", logits.shape)
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#
#
# device = torch.device(device)
# x = tokens.to(device)
#
#
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits, _= model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=1)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=1)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)
#
# for i in range(num_return_sequences):
#     tokens_list = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens_list)
#     print(">", decoded)