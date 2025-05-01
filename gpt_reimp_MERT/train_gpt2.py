
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
class DataLoaderLite:
     def __init__(self, B, T):
         self.B = B
         self.T = T
 
         # at init load tokens from disk and store them in memory
         with open('input.txt', 'r') as f:
             text = f.read()
         enc = tiktoken.get_encoding('gpt2')
         tokens = enc.encode(text)
         self.tokens = torch.tensor(tokens)
         print(f"loaded {len(self.tokens)} tokens")
         print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
 
         # state
         self.current_position = 0
 
     def next_batch(self):
         B, T = self.B, self.T
         buf = self.tokens[self.current_position : self.current_position+B*T+1]
         x = (buf[:-1]).view(B, T) # inputs
         y = (buf[1:]).view(B, T) # targets
         # advance the position in the tensor
         self.current_position += B * T
         # if loading the next batch would be out of bounds, reset
         if self.current_position + (B * T + 1) > len(self.tokens):
             self.current_position = 0
         return x, y
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
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'input.txt')
with open(input_path, 'r', encoding='utf-8') as f:
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


# Train/Test Split için kod
# Train/Test Split için kod


# Metin üretme fonksiyonu - tiktoken problemi düzeltildi
# Metin üretme fonksiyonu - tüm hatalar düzeltildi
def generate(model, prompt, max_tokens=100, temperature=0.8, top_k=40):
    # Prompt'u tokenize et - özel tokenlere izin ver
    prompt_tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # EOS token - açık bir şekilde tanımla
    # GPT-2 için genellikle bu değer 50256'dır
    eos_token = 50256  # GPT-2'nin <|endoftext|> token ID'si

    # Metin üretme döngüsü
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Context boyutu kontrol et
            if x.size(1) > model.config.block_size:
                x = x[:, -model.config.block_size:]

            # Modelden çıktıları al
            logits, _ = model(x)

            # Son token için logits'leri al
            logits = logits[:, -1, :] / temperature

            # Top-k sampling uygula
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

            # Olasılık dağılımı oluştur
            probs = F.softmax(logits, dim=-1)

            # Bir sonraki token'ı örnekle
            next_token = torch.multinomial(probs, num_samples=1)

            # Üretilen token'ı kaydet
            generated_tokens.append(next_token.item())

            # Giriş dizisini güncelle
            x = torch.cat((x, next_token), dim=1)

            # EOS token'ı kontrol et
            if next_token.item() == eos_token:
                break

    # Token'ları decode et - allowed_special parametresi olmadan
    generated_text = enc.decode(generated_tokens)
    return prompt + generated_text


# Başlangıç metni
prompt = "The future of artificial intelligence is"

# Generate text with different temperature values
print("=== Generated Text (Temperature = 0.7) ===")
output1 = generate(model, prompt, max_tokens=50, temperature=0.7)
print(output1)
print("\n")

print("=== Generated Text (Temperature = 1.0) ===")
output2 = generate(model, prompt, max_tokens=50, temperature=1.0)
print(output2)
print("\n")

print("=== Generated Text (Temperature = 0.5) ===")
output3 = generate(model, prompt, max_tokens=50, temperature=0.5, top_k=20)
print(output3)



#
# import random
#
#
# # Veriyi karıştırıp train/test olarak ayırma
# def train_test_split(tokens, train_ratio=0.8):
#     random.seed(42)
#     n = len(tokens)
#     split_idx = int(n * train_ratio)
#
#     # Veriyi yeterince büyük parçalara bölebilmek için
#     train_tokens = tokens[:split_idx]
#     test_tokens = tokens[split_idx:]
#
#     return train_tokens, test_tokens
#
#
# # Batch oluşturma fonksiyonu - daha güvenli versiyonu
# def get_batch(data, batch_size, seq_length):
#     # Veri setinin boyutunu kontrol et
#     if len(data) <= seq_length + 1:
#         raise ValueError(f"Veri boyutu çok küçük. En az {seq_length + 2} token gerekli")
#
#     max_idx = len(data) - seq_length - 1
#     if max_idx <= 0:
#         raise ValueError(f"Veri boyutu çok küçük. En az {seq_length + 2} token gerekli")
#
#     # Rastgele başlangıç noktası seç
#     idx = random.randint(0, max_idx)
#
#     # Input ve target oluştur
#     x = torch.tensor(data[idx:idx + seq_length], dtype=torch.long).unsqueeze(0)
#     y = torch.tensor(data[idx + 1:idx + 1 + seq_length], dtype=torch.long).unsqueeze(0)
#
#     # Batch_size > 1 ise ve yeterli veri varsa birden fazla örnek oluştur
#     if batch_size > 1:
#         for _ in range(min(batch_size - 1, max_idx // seq_length)):
#             idx = random.randint(0, max_idx)
#             x_temp = torch.tensor(data[idx:idx + seq_length], dtype=torch.long).unsqueeze(0)
#             y_temp = torch.tensor(data[idx + 1:idx + 1 + seq_length], dtype=torch.long).unsqueeze(0)
#             x = torch.cat([x, x_temp], dim=0)
#             y = torch.cat([y, y_temp], dim=0)
#
#     return x.to(device), y.to(device)
#
#
# # Tokenları yükle
# with open('input.txt', 'r') as f:
#     text = f.read()
#
# # Tüm metni kullan
# tokens = enc.encode(text)
#
# # Eğitim parametreleri
# batch_size = 2  # Veri seti küçük olduğu için küçük bir batch size kullan
# seq_length = 16  # Daha kısa sekans uzunluğu
# learning_rate = 1e-3
# epochs = 50
# eval_interval = 5
#
# # Train ve test verilerini ayırma
# train_ratio = 0.8
# train_tokens, test_tokens = train_test_split(tokens, train_ratio)
# print(f"Train veri boyutu: {len(train_tokens)}, Test veri boyutu: {len(test_tokens)}")
#
# # Test verisi için parametreleri ayarla
# if len(test_tokens) < batch_size * seq_length + 1:
#     print("Test veri seti çok küçük. Test için daha küçük parametreler kullanılacak.")
#     test_batch_size = 1
#     test_seq_length = min(16, len(test_tokens) - 2)
# else:
#     test_batch_size = batch_size
#     test_seq_length = seq_length
#
# print(f"Train parametreleri: batch_size={batch_size}, seq_length={seq_length}")
# print(f"Test parametreleri: batch_size={test_batch_size}, seq_length={test_seq_length}")
#
# # Model, optimizer ve loss
# model = GPT(GPTConfig())
# model = model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# # Eğitim ve test
# train_losses = []
# test_losses = []
# test_epochs = []  # Test loss hesaplanan epoch'ları takip etmek için
#
# try:
#     for epoch in range(epochs):
#         # Eğitim adımı
#         model.train()
#         try:
#             x_train, y_train = get_batch(train_tokens, batch_size, seq_length)
#             optimizer.zero_grad()
#             logits, train_loss = model(x_train, y_train)
#             train_loss.backward()
#             optimizer.step()
#             train_losses.append(train_loss.item())
#
#             # Test adımı
#             if epoch % eval_interval == 0 or epoch == epochs - 1:
#                 model.eval()
#                 with torch.no_grad():
#                     try:
#                         x_test, y_test = get_batch(test_tokens, test_batch_size, test_seq_length)
#                         _, test_loss = model(x_test, y_test)
#                         test_losses.append(test_loss.item())
#                         test_epochs.append(epoch)
#                         print(f"Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
#                     except ValueError as e:
#                         print(f"Test veri hatası: {e}")
#                         print(f"Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, Test yapılamadı")
#             else:
#                 print(f"Epoch: {epoch}, Train Loss: {train_loss.item():.4f}")
#         except ValueError as e:
#             print(f"Veri hatası: {e}")
#             continue
# except Exception as e:
#     print(f"Eğitim sırasında hata: {e}")
#
# # Eğer test_losses boşsa, çizimi atlayacak bir kontrol ekleyelim
# if len(test_losses) > 0:
#     # Sonuçları görselleştirme
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
#
#     plt.subplot(1, 2, 2)
#     plt.plot(test_epochs, test_losses, marker='o', label='Test Loss')
#     plt.title("Test Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig('training_results.png')  # Grafiği kaydet
#     plt.show()
# else:
#     print("Test loss hesaplanamadı, grafik oluşturulamıyor.")
#
#
# # Modeli test etmek - metin üretme
# def generate_text(model, prefix_tokens, max_new_tokens=50, temperature=1.0, top_k=50):
#     model.eval()
#     x = torch.tensor(prefix_tokens, dtype=torch.long).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             # Sadece son seq_length tokenları kullan
#             x_cond = x[:, -model.config.block_size:] if x.size(1) > model.config.block_size else x
#             logits, _ = model(x_cond)
#             logits = logits[:, -1, :] / temperature
#
#             # Top-k sampling
#             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             logits[logits < v[:, [-1]]] = float('-inf')
#             probs = F.softmax(logits, dim=-1)
#
#             # Sample
#             next_token = torch.multinomial(probs, num_samples=1)
#             x = torch.cat([x, next_token], dim=1)
#
#     generated_tokens = x[0].tolist()
#     return enc.decode(generated_tokens)
#
#
# # Veri setinden bir başlangıç metni al
# try:
#     # Test verisi yeterince büyükse
#     if len(test_tokens) >= 10:
#         test_prefix = test_tokens[:10]
#     else:
#         test_prefix = test_tokens[:len(test_tokens) // 2]
#
#     generated_text = generate_text(model, test_prefix)
#     print("\nÜretilen metin:")
#     print(generated_text)
# except Exception as e:
#     print(f"Metin üretme sırasında hata: {e}")