"""
Minimal implementation of an image diffusion model on MNIST 

• Training: python src/mnist.py --train
• Sampling: python src/mnist.py --sample --ckpt ckpt.pth
"""
import argparse
import math
import os
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from .utils import load_checkpoint, save_checkpoint, get_vertex_checkpoint_path, get_samples_dir, save_samples


def linear_beta_schedule(timesteps: int, start=1e-4, end=2e-2):
    """Linear schedule from Ho et al. 2020."""
    return torch.linspace(start, end, timesteps)

timesteps = 1000  # Total diffusion steps
betas = linear_beta_schedule(timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise=None):
    """Diffuse the data for a given timestep t."""
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_acp = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_om_acp = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_acp * x_start + sqrt_om_acp * noise


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_emb = nn.Linear(1, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        time_bias = self.time_emb(t).view(t.shape[0], -1, 1, 1)
        h = h + time_bias
        h = F.relu(self.conv2(h))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rb1 = ResidualBlock( 1, 32)     # 1  → 32
        self.rb2 = ResidualBlock(32, 64)     # 32 → 64

        self.rb3 = ResidualBlock(64, 64)     # 64 → 64

        self.rb4 = ResidualBlock(96, 32)     # 96 → 32
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        t = (t.float() / timesteps).view(-1, 1, 1, 1)

        h1 = self.rb1(x, t)                  # (B, 32, H,  W)
        h2 = self.rb2(F.avg_pool2d(h1, 2), t)# (B, 64, H/2, W/2)
        h3 = self.rb3(h2, t)                 # (B, 64, H/2, W/2)

        h4 = F.interpolate(h3, scale_factor=2, mode="nearest")  # (B, 64, H, W)
        h4 = torch.cat([h4, h1], dim=1)       # (B, 96, H, W)  <- skip connection
        h4 = self.rb4(h4, t)                  # (B, 32, H, W)

        return self.out(h4)                   # (B,  1, H, W)


@contextmanager
def eval_mode(module: nn.Module):
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)

def sample_images(model: nn.Module, device: str, epoch: int,
                  n_samples: int = 25, outdir: str = "samples"):
    samples_dir = get_samples_dir(outdir)
    
    with eval_mode(model), torch.no_grad():
        x = torch.randn(n_samples, 1, 28, 28, device=device)
        for i in reversed(range(timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = p_sample(model, x, t)
        x = (x.clamp(-1, 1) + 1) / 2
        
        # Save using torchvision's save_image to get a grid
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            utils.save_image(x, tmp.name, nrow=int(math.sqrt(n_samples)))
            
            # Read the image bytes and save using our utils
            with open(tmp.name, "rb") as f:
                image_bytes = f.read()
            
            if isinstance(samples_dir, str):
                sample_path = f"{samples_dir}/epoch_{epoch:03d}.png"
            else:
                sample_path = samples_dir / f"epoch_{epoch:03d}.png"
            save_samples(image_bytes, sample_path, mode="wb")
            os.unlink(tmp.name)
            
    print(f"[epoch {epoch}] saved samples to {sample_path}")

def train(model: nn.Module,
          device: str,
          epochs: int = 5,
          batch_size: int = 128,
          lr: float = 1e-3,
          ckpt_path: str = "ckpt.pth",
          sample_every_epoch: bool = True,
          samples_per_epoch: int = 25):
    
    ckpt_path = get_vertex_checkpoint_path("image-model.pth") if "AIP_MODEL_DIR" in os.environ else ckpt_path

    ds = datasets.MNIST(
        "./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_noisy = q_sample(x, t, noise)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if sample_every_epoch:
            sample_images(model, device, epoch + 1, samples_per_epoch)

    save_checkpoint(model.state_dict(), ckpt_path)

def p_sample(model, x, t):
    """Perform one reverse step."""
    betas_t = betas[t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    sqrt_recip_alphas_t = (1.0 / torch.sqrt(alphas[t]))[:, None, None, None]

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t)
    )
    if t[0] == 0:
        return model_mean
    noise = torch.randn_like(x)
    posterior_var = betas[t][:, None, None, None]
    return model_mean + torch.sqrt(posterior_var) * noise


def sample(model: nn.Module, device: str, n_samples=25, ckpt_path="ckpt.pth", outdir="samples"):
    model.load_state_dict(load_checkpoint(ckpt_path, device))
    model.eval()
    
    samples_dir = get_samples_dir(outdir)
    
    with torch.no_grad():
        x = torch.randn(n_samples, 1, 28, 28, device=device)
        for i in tqdm(reversed(range(timesteps)), desc="Sampling"):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = p_sample(model, x, t)
        x = (x.clamp(-1, 1) + 1) / 2  # back to [0,1]
        
        # Save using torchvision's save_image to get a grid
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            utils.save_image(x, tmp.name, nrow=int(math.sqrt(n_samples)))
            
            # Read the image bytes and save using our utils
            with open(tmp.name, "rb") as f:
                image_bytes = f.read()
            
            if isinstance(samples_dir, str):
                sample_path = f"{samples_dir}/samples.png"
            else:
                sample_path = samples_dir / "samples.png"
            save_samples(image_bytes, sample_path, mode="wb")
            os.unlink(tmp.name)
            
        print(f"Saved samples to {sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--sample", action="store_true", help="Generate samples")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt", type=str, default=get_vertex_checkpoint_path("image-model.pth") if "AIP_MODEL_DIR" in os.environ else "ckpt.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move global diffusion tensors to device
    # global sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, alphas
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)
    
    model = SimpleUNet().to(device)

    if args.train:
        train(model, device, epochs=args.epochs, batch_size=args.batch_size, ckpt_path=args.ckpt)
    if args.sample:
        sample(model, device, ckpt_path=args.ckpt)

    if not args.train and not args.sample:
        print("Nothing to do. Pass --train or --sample.")