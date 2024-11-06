import torch
import torch.nn as nn
import torch.nn.functional as F
from triformer import TritonLayerNorm, TritonSoftmax, TritonDropout

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name()
    print(f"Using GPU: {gpu_name}")
    
    if "A100" in gpu_name or "3090" in gpu_name or "A40" in gpu_name or "A6000" in gpu_name or "A10" in gpu_name:
        print("Detected Ampere or newer GPU. Enabling FlashAttention.")
        torch.backends.cuda.enable_flash_sdp(True)  
        torch.backends.cuda.enable_mem_efficient_sdp(False) 
    else:
        print("Non-Ampere GPU detected. Enabling Memory-Efficient Attention.")
        torch.backends.cuda.enable_flash_sdp(False)  
        torch.backends.cuda.enable_mem_efficient_sdp(True)  
else:
    print("CUDA is not available. Using default PyTorch C++ attention backend.")

def create_causal_mask(seq_len, device):
    """
    Creates a causal (autoregressive) mask
    Each position can only attend to previous positions
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_flash_attention:bool=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.use_flash_attention = use_flash_attention
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Split into heads and reshape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash_attention:
            if mask is not None:
                # Expand mask for multiple heads
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            
        else:
            scale = k.size(-1) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                mask = mask.unsqueeze(1)  
                attn = attn.masked_fill(~mask, float('-inf'))
            
            attn = TritonSoftmax(is_causal=True)(attn)
            attn_output = torch.matmul(attn, v)
        
        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        return self.out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()  
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        if self.training:
            seed = torch.randint(0, 2**31-1, (1,)).item()
            x = TritonDropout.apply(x, self.dropout_prob, seed)
        x = self.linear2(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_prob=0.1, use_flash_attention:bool=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, use_flash_attention=use_flash_attention)
        self.ln1 = TritonLayerNorm(d_model)
        self.ln2 = TritonLayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout_prob)
        self.dropout_prob = dropout_prob
        
    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = x + attn_out  
        x = self.ln1(x)
        ff_out = self.ff(x)
        x = x + ff_out  
        x = self.ln2(x)
        return x

class GPT2(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        max_seq_len,
        d_model=768,      
        num_layers=12,    
        num_heads=12,     
        d_ff=3072,        
        dropout_prob=0.1,
        use_flash_attention:bool=False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, num_heads, d_ff, dropout_prob, use_flash_attention=use_flash_attention)
            for _ in range(num_layers)
        ])
        self.ln_f = TritonLayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)
        x = token_embeddings + position_embeddings
        mask = create_causal_mask(x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
