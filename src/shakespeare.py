import argparse
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from .utils import load_checkpoint, save_checkpoint, get_vertex_checkpoint_path, get_samples_dir, save_samples

HF_TOKEN = os.getenv("HF_TOKEN")


T = 1_000  # number of diffusion steps

def linear_beta_schedule(timesteps: int, start=1e-4, end=2e-2):
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

def q_sample(x0: torch.Tensor, t: torch.Tensor, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    b = x0.shape[0]
    device = x0.device
    sqrt_acp = sqrt_alphas_cumprod[t].to(device).view(b, 1, 1)
    sqrt_om = sqrt_one_minus_alphas_cumprod[t].to(device).view(b, 1, 1)
    return sqrt_acp * x0 + sqrt_om * noise

class LearnedRounding(nn.Module):
    """Learned rounding function to convert embeddings to token probabilities."""
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.decoder = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, embeddings):
        """Convert embeddings to token logits.
        
        Args:
            embeddings: (B, L, embed_dim)
        
        Returns:
            logits: (B, L, vocab_size)
        """
        return self.decoder(embeddings)


class TinyTransformer(nn.Module):
    def __init__(self, dim, n_heads=4, depth=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.time_emb = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_scaled = (t.float() / T).unsqueeze(-1)  # (B,1)
        time_bias = self.time_emb(t_scaled).unsqueeze(1)  # (B,1,dim)
        x = x + time_bias
        return self.encoder(x)

def load_text_dataset():
    """Return the raw Shakespeare corpus as a single string."""
    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)
    return "\n\n".join(ds['train']["text"] + ds['test']["text"] + ds['validation']["text"])


def tokenize_corpus(text: str, tokenizer, seq_len: int):
    """Tokenize full corpus *once* and slice into fixed‑length chunks.

    Args:
        text: full corpus as a single string
        tokenizer: HF tokenizer (we disable special tokens)
        seq_len: target sequence length

    Returns:
        Tensor of shape (N_chunks, seq_len) with `dtype long`.
    """
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors="pt",
    ).input_ids.squeeze(0)  # (total_len,)
    total_len = ids.size(0)
    n_chunks = total_len // seq_len
    ids = ids[: n_chunks * seq_len]  # drop remainder
    chunks = ids.view(n_chunks, seq_len)
    return chunks


def train(
    model,
    rounding_fn,
    embed_matrix,
    data_loader,
    device,
    ckpt_path="text_ckpt.pth",
    epochs=1,
    lr=1e-4,
    rounding_weight=1.0,
):
    
    optim = torch.optim.AdamW(list(model.parameters()) + list(rounding_fn.parameters()), lr=lr)
    for epoch in range(epochs):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for token_ids in pbar:
            token_ids = token_ids.to(device)
            x0 = embed_matrix[token_ids]  # (B, L, dim)
            t = torch.randint(0, T, (x0.shape[0],), device=device).long()
            noise = torch.randn_like(x0)
            x_noisy = q_sample(x0, t, noise)
            noise_pred = model(x_noisy, t)
            
            # Diffusion loss (denoising objective)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            
            # Rounding loss (token prediction objective)
            logits = rounding_fn(x0)  # (B, L, vocab_size)
            rounding_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1))
            
            # Combined loss
            total_loss = diffusion_loss + rounding_weight * rounding_loss

            optim.zero_grad(); total_loss.backward(); optim.step()
            pbar.set_postfix(
                diff_loss=diffusion_loss.item(), 
                round_loss=rounding_loss.item(), 
                total=total_loss.item()
            )
    
    ckpt_path = get_vertex_checkpoint_path("text-model.pth") if "AIP_MODEL_DIR" in os.environ else ckpt_path

    print(f"✔ Saving checkpoint to {ckpt_path}...")
    checkpoint = {
        'diffusion_model': model.state_dict(),
        'rounding_fn': rounding_fn.state_dict()
    }
    save_checkpoint(checkpoint, ckpt_path)

def p_sample(model, x, t):
    beta_t = betas[t].view(-1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    sqrt_recip_alpha = (1.0 / torch.sqrt(alphas[t])).view(-1, 1, 1)

    model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus * model(x, t))
    if t[0] == 0:
        return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(beta_t) * noise


def sample(
    model,
    rounding_fn,
    embed_matrix,
    tokenizer,
    device,
    n_samples=4,
    seq_len=128,
    use_learned_rounding=True,
):
    model.eval()
    rounding_fn.eval()
    samples_dir = get_samples_dir("samples")
    
    with torch.no_grad():
        x = torch.randn(n_samples, seq_len, embed_matrix.shape[1], device=device)
        for i in tqdm(reversed(range(T)), desc="Sampling"):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = p_sample(model, x, t)
        
        if use_learned_rounding:
            # Use learned rounding function
            logits = rounding_fn(x)  # (B, L, V)
            tokens = logits.argmax(dim=-1)  # (B, L)
        else:
            # Fall back to cosine similarity (original method)
            emb_norm = F.normalize(embed_matrix, dim=1)  # (V, dim)
            x_norm = F.normalize(x, dim=2)               # (B, L, dim)
            sims = torch.matmul(x_norm, emb_norm.T)      # (B, L, V)
            tokens = sims.argmax(dim=-1)                 # (B, L)
        
        texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        
        for i, text in enumerate(texts):
            print(text)
            # Use string concatenation for GCS paths to avoid Path() corruption
            if isinstance(samples_dir, str) and samples_dir.startswith("gs://"):
                sample_path = f"{samples_dir}/sample_{i}.txt"
            else:
                sample_path = Path(samples_dir) / f"sample_{i}.txt"
            save_samples(text, sample_path)
            print(f"✔ Wrote {sample_path}")
        
        return texts


def sample_diffusion_embeddings(model, embed_dim, device, n, seq_len):
    """Generate *pure* embeddings z using only the diffusion model."""
    x = torch.randn(n, seq_len, embed_dim, device=device)
    model.eval()
    with torch.no_grad():
        for i in tqdm(reversed(range(T)), desc="Diffusing emb."):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = p_sample(model, x, t)
    return x  # (B,L,dim)


def guided_generate(
    base_lm: nn.Module,
    rounding_fn: nn.Module,
    tokenizer,
    embed_matrix: torch.Tensor,
    diff_z: torch.Tensor,
    alpha: float = 0.5,
    max_len: int = 128,
    temperature: float = 1.0,
    use_learned_rounding: bool = True,
):
    """One batch (size = diff_z.size(0)) of guided generation."""

    device = diff_z.device
    B, L, dim = diff_z.shape
    input_ids = torch.full((B, 1), tokenizer.bos_token_id or tokenizer.eos_token_id, device=device, dtype=torch.long)

    for pos in range(L):
        outputs = base_lm(input_ids)
        ar_logits = outputs.logits[:, -1, :] / temperature   # (B,V)

        if use_learned_rounding:
            # Use learned rounding function for diffusion logits
            z_pos = diff_z[:, pos:pos+1, :]  # (B, 1, dim)
            diff_logits = rounding_fn(z_pos).squeeze(1) / temperature  # (B, V)
        else:
            # Fall back to cosine similarity
            emb_norm = F.normalize(embed_matrix, dim=1)             # (V,dim)
            z_norm = F.normalize(diff_z[:, pos, :], dim=1)       # (B,dim)
            diff_logits = torch.matmul(z_norm, emb_norm.T) / temperature  # (B,V)

        mixed_logits = (1 - alpha) * ar_logits + alpha * diff_logits
        next_id = torch.argmax(mixed_logits, dim=-1, keepdim=True)    # greedy
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true", help="plain diffusion sample")
    parser.add_argument("--guided_sample", action="store_true", help="AR + diffusion guidance")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--ckpt", type=str, default="gs://text-diffusion/diffusion/outputs/model/text-model.pth" if "AIP_MODEL_DIR" in os.environ else "text_ckpt.pth")
    parser.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--rounding_weight", type=float, default=1.0, help="Weight for learned rounding loss")
    parser.add_argument("--use_cosine_fallback", action="store_true", help="Use cosine similarity instead of learned rounding")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    lm_model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    embed_matrix = lm_model.get_input_embeddings().weight.detach().to(device)
    dim = embed_matrix.size(1)
    vocab_size = embed_matrix.size(0)

    diff_model = TinyTransformer(dim).to(device)
    rounding_fn = LearnedRounding(dim, vocab_size).to(device)

    if args.train:
        raw = load_text_dataset()
        chunks = tokenize_corpus(raw, tokenizer, args.seq_len)
        dl = DataLoader(chunks, batch_size=args.batch_size, shuffle=True)
        train(diff_model, rounding_fn, embed_matrix, dl, device, args.ckpt, 
              epochs=args.epochs, rounding_weight=args.rounding_weight)

    if args.sample:
        checkpoint = load_checkpoint(args.ckpt, device)
        if isinstance(checkpoint, dict) and 'diffusion_model' in checkpoint:
            # New checkpoint format with both models
            diff_model.load_state_dict(checkpoint['diffusion_model'])
            rounding_fn.load_state_dict(checkpoint['rounding_fn'])
        else:
            # Old checkpoint format (diffusion model only)
            diff_model.load_state_dict(checkpoint)
            print("Warning: Using old checkpoint format without learned rounding. Falling back to cosine similarity.")
            args.use_cosine_fallback = True
        
        texts = sample(diff_model, rounding_fn, embed_matrix, tokenizer, device, 
                      args.n, args.seq_len, use_learned_rounding=not args.use_cosine_fallback)

    if args.guided_sample:
        checkpoint = load_checkpoint(args.ckpt, device)
        if isinstance(checkpoint, dict) and 'diffusion_model' in checkpoint:
            # New checkpoint format with both models
            diff_model.load_state_dict(checkpoint['diffusion_model'])
            rounding_fn.load_state_dict(checkpoint['rounding_fn'])
        else:
            # Old checkpoint format (diffusion model only)
            diff_model.load_state_dict(checkpoint)
            print("Warning: Using old checkpoint format without learned rounding. Falling back to cosine similarity.")
            args.use_cosine_fallback = True
            
        z = sample_diffusion_embeddings(diff_model, dim, device, args.n, args.seq_len)
        texts = guided_generate(lm_model, rounding_fn, tokenizer, embed_matrix, z, 
                               alpha=args.alpha, max_len=args.seq_len, 
                               use_learned_rounding=not args.use_cosine_fallback)
        samples_dir = get_samples_dir("samples")
        for i, text in enumerate(texts):
            # Use string concatenation for GCS paths to avoid Path() corruption
            if isinstance(samples_dir, str) and samples_dir.startswith("gs://"):
                sample_path = f"{samples_dir}/guided_sample_{i}.txt"
            else:
                sample_path = Path(samples_dir) / f"guided_sample_{i}.txt"
            save_samples(text, sample_path)
            print(f"✔ Wrote {sample_path}")

    if not (args.train or args.sample or args.guided_sample):
        print("Nothing to do. Try --train or --guided_sample.")