import argparse
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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

class LearnedEmbedding(nn.Module):
    """Custom learnable embedding space for diffusion."""
    def __init__(self, vocab_size, embed_dim, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embedding matrix
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        if pretrained_embeddings is not None:
            # Initialize with pre-trained embeddings if provided
            if pretrained_embeddings.size(1) != embed_dim:
                # Project to desired dimension if different
                projection = nn.Linear(pretrained_embeddings.size(1), embed_dim, bias=False).to(pretrained_embeddings.device)
                with torch.no_grad():
                    projected = projection(pretrained_embeddings)
                    self.embeddings.weight.copy_(projected)
            else:
                # Use pre-trained embeddings directly
                self.embeddings.weight.data.copy_(pretrained_embeddings)
        else:
            # Random initialization
            nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids):
        """Convert token IDs to embeddings.
        
        Args:
            token_ids: (B, L) tensor of token indices
            
        Returns:
            embeddings: (B, L, embed_dim) tensor of embeddings
        """
        return self.embeddings(token_ids)
    
    def get_embedding_matrix(self):
        """Get the full embedding matrix for decoding."""
        return self.embeddings.weight  # (vocab_size, embed_dim)


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
    def __init__(self, dim, n_heads=4, depth=3, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.time_emb = nn.Linear(1, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_scaled = (t.float() / T).unsqueeze(-1)  # (B,1)
        time_bias = self.time_emb(t_scaled).unsqueeze(1)  # (B,1,dim)
        x = x + time_bias
        x = self.dropout(x)
        return self.encoder(x)

def load_text_dataset():
    """Return the raw Shakespeare corpus as a single string."""
    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)
    return "\n\n".join(ds['train']["text"] + ds['test']["text"] + ds['validation']["text"])


def tokenize_corpus(text: str, tokenizer, seq_len: int, val_split=0.1):
    """Tokenize full corpus *once* and slice into fixed‑length chunks with train/val split.

    Args:
        text: full corpus as a single string
        tokenizer: HF tokenizer (we disable special tokens)
        seq_len: target sequence length
        val_split: fraction of data to use for validation

    Returns:
        Tuple of (train_chunks, val_chunks) tensors of shape (N_chunks, seq_len) with `dtype long`.
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
    
    # Split into train/val
    n_val = int(n_chunks * val_split)
    n_train = n_chunks - n_val
    train_chunks, val_chunks = random_split(chunks, [n_train, n_val])
    
    return train_chunks, val_chunks


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=0):
    """Cosine annealing learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def dynamic_rounding_weight_schedule(epoch, total_epochs, initial_weight=1.0, final_weight=0.1):
    """Decay rounding weight over training to prevent overfitting to token prediction."""
    progress = epoch / total_epochs
    return initial_weight * (1 - progress) + final_weight * progress

def train(
    model,
    rounding_fn,
    embedding_fn,
    data_loader,
    val_loader,
    device,
    ckpt_path="text_ckpt.pth",
    epochs=1,
    lr=1e-4,
    weight_decay=1e-4,
    rounding_weight=1.0,
    use_learned_embeddings=True,
    patience=5,
    use_lr_scheduling=True,
    warmup_steps=100,
):
    
    # Include embedding parameters in optimization if using learned embeddings
    params = list(model.parameters()) + list(rounding_fn.parameters())
    if use_learned_embeddings:
        params += list(embedding_fn.parameters())
    
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduling
    total_steps = len(data_loader) * epochs
    if use_lr_scheduling:
        scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        rounding_fn.train()
        if use_learned_embeddings:
            embedding_fn.train()
        
        # Dynamic rounding weight scheduling
        current_rounding_weight = dynamic_rounding_weight_schedule(epoch, epochs, rounding_weight)
        
        train_losses = {'diff': 0, 'round': 0, 'total': 0}
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")
        
        for step, token_ids in enumerate(pbar):
            token_ids = token_ids.to(device)
            
            # Get embeddings (either learned or pre-trained)
            if use_learned_embeddings:
                x0 = embedding_fn(token_ids)  # (B, L, dim)
            else:
                x0 = embedding_fn[token_ids]  # (B, L, dim) - direct indexing for pre-trained
            
            t = torch.randint(0, T, (x0.shape[0],), device=device).long()
            noise = torch.randn_like(x0)
            x_noisy = q_sample(x0, t, noise)
            noise_pred = model(x_noisy, t)
            
            # Diffusion loss (denoising objective)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            
            # Rounding loss (token prediction objective)
            logits = rounding_fn(x0)  # (B, L, vocab_size)
            rounding_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1))
            
            # Combined loss with dynamic weighting
            total_loss = diffusion_loss + current_rounding_weight * rounding_loss

            optim.zero_grad()
            total_loss.backward()
            optim.step()
            
            if use_lr_scheduling:
                scheduler.step()
            
            # Track losses
            train_losses['diff'] += diffusion_loss.item()
            train_losses['round'] += rounding_loss.item()
            train_losses['total'] += total_loss.item()
            
            pbar.set_postfix(
                diff_loss=diffusion_loss.item(), 
                round_loss=rounding_loss.item(), 
                total=total_loss.item(),
                rw=current_rounding_weight,
                lr=optim.param_groups[0]['lr']
            )
        
        # Validation phase
        model.eval()
        rounding_fn.eval()
        if use_learned_embeddings:
            embedding_fn.eval()
        
        val_losses = {'diff': 0, 'round': 0, 'total': 0}
        with torch.no_grad():
            for token_ids in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                token_ids = token_ids.to(device)
                
                if use_learned_embeddings:
                    x0 = embedding_fn(token_ids)
                else:
                    x0 = embedding_fn[token_ids]
                
                t = torch.randint(0, T, (x0.shape[0],), device=device).long()
                noise = torch.randn_like(x0)
                x_noisy = q_sample(x0, t, noise)
                noise_pred = model(x_noisy, t)
                
                diffusion_loss = F.mse_loss(noise_pred, noise)
                logits = rounding_fn(x0)
                rounding_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1))
                total_loss = diffusion_loss + current_rounding_weight * rounding_loss
                
                val_losses['diff'] += diffusion_loss.item()
                val_losses['round'] += rounding_loss.item()
                val_losses['total'] += total_loss.item()
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= len(data_loader)
            val_losses[key] /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train: diff={train_losses['diff']:.4f}, round={train_losses['round']:.4f}, total={train_losses['total']:.4f}")
        print(f"  Val:   diff={val_losses['diff']:.4f}, round={val_losses['round']:.4f}, total={val_losses['total']:.4f}")
        print(f"  Rounding weight: {current_rounding_weight:.3f}")
        
        # Early stopping check
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            # Save best checkpoint
            best_ckpt_path = ckpt_path.replace('.pth', '_best.pth')
            checkpoint = {
                'diffusion_model': model.state_dict(),
                'rounding_fn': rounding_fn.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss
            }
            if use_learned_embeddings:
                checkpoint['embedding_fn'] = embedding_fn.state_dict()
            save_checkpoint(checkpoint, best_ckpt_path)
            print(f"  New best validation loss! Saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Save final checkpoint
    final_ckpt_path = get_vertex_checkpoint_path("text-model.pth") if "AIP_MODEL_DIR" in os.environ else ckpt_path
    print(f"✔ Saving final checkpoint to {final_ckpt_path}...")
    final_checkpoint = {
        'diffusion_model': model.state_dict(),
        'rounding_fn': rounding_fn.state_dict(),
        'epoch': epochs,
        'final_training': True
    }
    
    # Save embedding function if it's learned
    if use_learned_embeddings:
        final_checkpoint['embedding_fn'] = embedding_fn.state_dict()
    
    save_checkpoint(final_checkpoint, final_ckpt_path)

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
    embedding_fn,
    tokenizer,
    device,
    n_samples=4,
    seq_len=128,
    use_learned_rounding=True,
    use_learned_embeddings=True,
    embed_dim=None,
):
    model.eval()
    rounding_fn.eval()
    if use_learned_embeddings:
        embedding_fn.eval()
    
    samples_dir = get_samples_dir("samples")
    
    with torch.no_grad():
        # Determine embedding dimension
        if embed_dim is None:
            if use_learned_embeddings:
                embed_dim = embedding_fn.embed_dim
            else:
                embed_dim = embedding_fn.shape[1]  # Pre-trained embedding matrix
        
        x = torch.randn(n_samples, seq_len, embed_dim, device=device)
        for i in tqdm(reversed(range(T)), desc="Sampling"):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = p_sample(model, x, t)
        
        if use_learned_rounding:
            # Use learned rounding function
            logits = rounding_fn(x)  # (B, L, V)
            tokens = logits.argmax(dim=-1)  # (B, L)
        else:
            # Fall back to cosine similarity (original method)
            if use_learned_embeddings:
                embed_matrix = embedding_fn.get_embedding_matrix()  # (V, dim)
            else:
                embed_matrix = embedding_fn  # Pre-trained embedding matrix
            
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
    embedding_fn,
    diff_z: torch.Tensor,
    alpha: float = 0.5,
    max_len: int = 128,
    temperature: float = 1.0,
    use_learned_rounding: bool = True,
    use_learned_embeddings: bool = True,
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
            if use_learned_embeddings:
                embed_matrix = embedding_fn.get_embedding_matrix()  # (V, dim)
            else:
                embed_matrix = embedding_fn  # Pre-trained embedding matrix
            
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
    parser.add_argument("--use_learned_embeddings", action="store_true", help="Use custom learned embedding space")
    parser.add_argument("--embed_dim", type=int, default=None, help="Custom embedding dimension (uses pre-trained dim if not specified)")
    parser.add_argument("--init_from_pretrained", action="store_true", help="Initialize learned embeddings from pre-trained weights")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for regularization")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--use_lr_scheduling", action="store_true", default=True, help="Use cosine learning rate scheduling")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduling")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
    pretrained_embed_matrix = lm_model.get_input_embeddings().weight.detach().to(device)
    pretrained_dim = pretrained_embed_matrix.size(1)
    vocab_size = pretrained_embed_matrix.size(0)

    # Determine embedding configuration
    if args.use_learned_embeddings:
        embed_dim = args.embed_dim if args.embed_dim is not None else pretrained_dim
        init_embeddings = pretrained_embed_matrix if args.init_from_pretrained else None
        embedding_fn = LearnedEmbedding(vocab_size, embed_dim, init_embeddings).to(device)
        print(f"Using learned embeddings (dim={embed_dim}, init_from_pretrained={args.init_from_pretrained})")
    else:
        embed_dim = pretrained_dim
        embedding_fn = pretrained_embed_matrix  # Direct tensor for indexing
        print(f"Using pre-trained embeddings (dim={embed_dim})")

    diff_model = TinyTransformer(embed_dim, dropout=args.dropout).to(device)
    rounding_fn = LearnedRounding(embed_dim, vocab_size).to(device)

    if args.train:
        raw = load_text_dataset()
        train_chunks, val_chunks = tokenize_corpus(raw, tokenizer, args.seq_len, args.val_split)
        train_dl = DataLoader(train_chunks, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(val_chunks, batch_size=args.batch_size, shuffle=False)
        
        print(f"Training on {len(train_chunks)} chunks, validating on {len(val_chunks)} chunks")
        
        train(diff_model, rounding_fn, embedding_fn, train_dl, val_dl, device, args.ckpt, 
              epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
              rounding_weight=args.rounding_weight, use_learned_embeddings=args.use_learned_embeddings,
              patience=args.patience, use_lr_scheduling=args.use_lr_scheduling, 
              warmup_steps=args.warmup_steps)

    if args.sample:
        checkpoint = load_checkpoint(args.ckpt, device)
        if isinstance(checkpoint, dict) and 'diffusion_model' in checkpoint:
            # New checkpoint format with multiple models
            diff_model.load_state_dict(checkpoint['diffusion_model'])
            rounding_fn.load_state_dict(checkpoint['rounding_fn'])
            
            # Load embedding function if available
            if args.use_learned_embeddings and 'embedding_fn' in checkpoint:
                embedding_fn.load_state_dict(checkpoint['embedding_fn'])
            elif args.use_learned_embeddings and 'embedding_fn' not in checkpoint:
                print("Warning: Learned embeddings requested but not found in checkpoint. Using pre-trained fallback.")
                args.use_learned_embeddings = False
                embedding_fn = pretrained_embed_matrix
        else:
            # Old checkpoint format (diffusion model only)
            diff_model.load_state_dict(checkpoint)
            print("Warning: Using old checkpoint format. Falling back to pre-trained embeddings and cosine similarity.")
            args.use_cosine_fallback = True
            args.use_learned_embeddings = False
            embedding_fn = pretrained_embed_matrix
        
        texts = sample(diff_model, rounding_fn, embedding_fn, tokenizer, device, 
                      args.n, args.seq_len, use_learned_rounding=not args.use_cosine_fallback,
                      use_learned_embeddings=args.use_learned_embeddings, embed_dim=embed_dim)

    if args.guided_sample:
        checkpoint = load_checkpoint(args.ckpt, device)
        if isinstance(checkpoint, dict) and 'diffusion_model' in checkpoint:
            # New checkpoint format with multiple models
            diff_model.load_state_dict(checkpoint['diffusion_model'])
            rounding_fn.load_state_dict(checkpoint['rounding_fn'])
            
            # Load embedding function if available
            if args.use_learned_embeddings and 'embedding_fn' in checkpoint:
                embedding_fn.load_state_dict(checkpoint['embedding_fn'])
            elif args.use_learned_embeddings and 'embedding_fn' not in checkpoint:
                print("Warning: Learned embeddings requested but not found in checkpoint. Using pre-trained fallback.")
                args.use_learned_embeddings = False
                embedding_fn = pretrained_embed_matrix
        else:
            # Old checkpoint format (diffusion model only)
            diff_model.load_state_dict(checkpoint)
            print("Warning: Using old checkpoint format. Falling back to pre-trained embeddings and cosine similarity.")
            args.use_cosine_fallback = True
            args.use_learned_embeddings = False
            embedding_fn = pretrained_embed_matrix
            
        z = sample_diffusion_embeddings(diff_model, embed_dim, device, args.n, args.seq_len)
        texts = guided_generate(lm_model, rounding_fn, tokenizer, embedding_fn, z, 
                               alpha=args.alpha, max_len=args.seq_len, 
                               use_learned_rounding=not args.use_cosine_fallback,
                               use_learned_embeddings=args.use_learned_embeddings)
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