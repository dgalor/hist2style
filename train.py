import os
import io
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="where data/ (~54 GB) and checkpoints/ are stored; default: the repository folder")
parser.add_argument("--experiment", type=str, default="Hist2Style", help="run name = checkpoint sub-folder")
parser.add_argument("--seed", type=int, default=420)
parser.add_argument("--max_epochs", type=int, default=1500)
parser.add_argument("--save_all", action="store_true", help="keep a checkpoint for every epoch (default: only the last)")
args = parser.parse_args()

BASE_DIR = args.base_dir

os.makedirs(os.path.join(BASE_DIR, "data/"), exist_ok=True)

flux_style_ds_srgb_tar = "https://berkeley.box.com/shared/static/utl9vmomwk0vg2l0ngusyl0ri5ob1way"
unsplash25k_512_srgb_tar = "https://berkeley.box.com/shared/static/7wnwm0lme880vmsmu4xespn5uoz0bbzl"

def download_and_extract(name, url):
    data_dir = os.path.join(BASE_DIR, "data")
    destination = os.path.join(data_dir, name)
    if os.path.isdir(destination):
        return

    archive = os.path.join(data_dir, f"{name}.tar")
    subprocess.run(["wget", "-O", archive, url], check=True)
    subprocess.run(["tar", "-xf", archive, "-C", data_dir], check=True)
    if not os.path.isdir(destination):
        raise RuntimeError(f"Archive did not create expected directory: {destination}")
    os.remove(archive)


os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(BASE_DIR, ".cache/torchinductor/")
os.environ["TORCH_COMPILE_CACHE_DIR"] = os.path.join(BASE_DIR, ".cache/torchcompile/")
os.environ["TRITON_CACHE_DIR"] = os.path.join(BASE_DIR, ".cache/torchtriton/")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.makedirs(os.path.join(BASE_DIR, ".cache/torchinductor/"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, ".cache/torchcompile/"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, ".cache/torchtriton/"), exist_ok=True)

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torch.nn as nn
from torchvision import models
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as pl
from collections import defaultdict
from utils.score_cache_io import load_score_cache

#grid_sampler_3d_backward_cuda does not have a deterministic implementation
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False) 
torch.backends.cuda.matmul.allow_tf32 = True
dtype = "32" #be careful with changing (I've observed issues in quality when using bf16)

experiment = args.experiment

device = "0,"

size = (256, 256)
bsize = 64
gridsize = 16
histogram_size = 128
griddim = 8
img_path = os.path.join(BASE_DIR, "data/unsplash25k_512_srgb/")
style_path = os.path.join(BASE_DIR, "data/flux_style_ds_srgb/")
checkpoint_dir = os.path.join(BASE_DIR, "checkpoints/", experiment)
score_cache_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "utils",
    "score_cache.csv.gz",
)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

_RGB2YCBCR = torch.tensor([
    [0.2126,  0.7152,  0.0722],
    [-0.1146, -0.3854,  0.5],
    [0.5,     -0.4542, -0.0458],
], dtype=torch.float32)
def srgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB (gamma-encoded) float image(s) in [0, 1]
    → Y′CbCr float in [0, 1].
    Shape (..., 3, H, W)
    """
    mat = _RGB2YCBCR.to(device=img.device, dtype=img.dtype)
    out = img.movedim(-3, -1) @ mat.T
    out = torch.cat([out[..., 0:1].clip(0.0, 1.0), out[..., 1:3].clip(-0.5, 0.5)], dim=-1)
    return out.movedim(-1, -3)

Image.MAX_IMAGE_PIXELS = None
class DS(Dataset):
    def __init__(self, train=True, idx=None):
        self.files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')])
        self.val_transforms = T.Resize(size)
        self.train_transforms = nn.Sequential(
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.75, 1.0)),
        )
        scores = load_score_cache(score_cache_path)
        self.scores_by_im = defaultdict(list)
        for (im_name, style_name), score in scores.items():
            if score > 0.5:
                self.scores_by_im[im_name].append((style_name, score))
        self.files = [f for f in self.files if len(self.scores_by_im[f.split("/")[-1].split(".")[0]]) > 5]
        if idx is not None:
            self.files = [self.files[i] for i in idx]
            self.scores_by_im = {im_name.split("/")[-1].split(".")[0]: self.scores_by_im[im_name.split("/")[-1].split(".")[0]] for im_name in self.files}

        scores_by_style = defaultdict(list)
        for im_name, style_names in self.scores_by_im.items():
            for style_name, score in style_names:
                scores_by_style[style_name].append((im_name, score))

        self.train = train

    def __getitem__(self, index):
        im_name = self.files[index].split("/")[-1].split(".")[0]

        style_names = [i[0] for i in self.scores_by_im[im_name]]
        assert len(style_names) > 0, f"No styles found for {im_name}"
        style_name, style_name_target = np.random.choice(style_names, size=2, replace=False)

        p = np.random.rand()
        if p < 0.25:
            fname_source = os.path.join(img_path, im_name + ".jpg")
            fname_gt = os.path.join(style_path, im_name, style_name_target + ".jpg")
        else:
            fname_source = os.path.join(style_path, im_name, style_name + ".jpg")
            if p < 0.5:
                fname_gt = os.path.join(img_path, im_name + ".jpg")
            else:
                fname_gt = os.path.join(style_path, im_name, style_name_target + ".jpg")

        g = np.random.rand()
        img1, img3 = Image.open(fname_source), Image.open(fname_gt)

        if g < 0.05:
            img1 = img1.convert("L").convert("RGB")
        elif g < 0.1:
            img3 = img3.convert("L").convert("RGB")

        img1 = TF.to_tensor(img1)
        img3 = TF.to_tensor(img3)

        if self.train:
            augmentation_seed = torch.randint(0, 2**31, ()).item()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(augmentation_seed)
                img1 = self.train_transforms(img1)
                torch.manual_seed(augmentation_seed)
                img3 = self.train_transforms(img3)
        else:
            img1 = self.val_transforms(img1)
            img3 = self.val_transforms(img3)
        return img1, img3

    def __len__(self):
        return len(self.files)

def make_dataloaders():
    train_idx, val_idx = random_split(
        range(len(DS(train=False))),
        [0.9, 0.1],
        generator=torch.Generator().manual_seed(0),
    )
    train_dataset = DS(train=True, idx=train_idx)
    val_dataset = DS(train=False, idx=val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bsize,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bsize,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    return train_loader, val_loader

def batch_histogram_1D(imgs: torch.Tensor, num_bins: int = histogram_size, density: bool = True, icpct: bool = True) -> torch.Tensor:
    B, D, H, W = imgs.shape
    if icpct:
        assert D == 3, "ICtCp has 3 channels"
        imgs = imgs.clone()
        imgs[:, 1:] = imgs[:, 1:] + 0.5

    imgs_flat = imgs.reshape(B, D, H * W)
    coords = (imgs_flat * num_bins).floor().clamp(0, num_bins - 1).long()

    hist = torch.zeros(B, D, num_bins, dtype=imgs.dtype, device=imgs.device)
    ones = torch.ones_like(coords, dtype=imgs.dtype)
    hist = hist.scatter_add(2, coords, ones)

    if density:
        hist = hist.float()
        hist_sum = hist.sum(dim=2, keepdim=True).clamp_min(1e-8)
        hist = hist / hist_sum
    return hist

class GridSampler(nn.Module):
    def __init__(self):
        super(GridSampler, self).__init__()
        self.pre_guide = LearnedSeparableGuide()
        self.L_weights = nn.Parameter(torch.zeros((1, 3, 1, 1), dtype=torch.float32))
    def forward(self, bilateral_grid, img_content):
        B, _, H, W = img_content.shape

        yy = (2*torch.arange(H, device=img_content.device).float() + 1) / H - 1
        xx = (2*torch.arange(W, device=img_content.device).float() + 1) / W - 1
        Y, X = torch.meshgrid(yy, xx, indexing='ij')
        x, y = X.view(1, H, W).expand(B, -1, -1), Y.view(1, H, W).expand(B, -1, -1)
        w = F.softmax(self.L_weights, dim=1)
        pre = self.pre_guide(img_content)
        z = (pre*w).sum(1)
        grid3d = torch.stack([x, y, (z-0.5)*2], dim=-1).view(B, 1, H, W, 3).clip(-1, 1)
        coeff = F.grid_sample(bilateral_grid, grid3d, mode="bilinear", align_corners=False, padding_mode="border")
        return coeff.squeeze(2)

class LayerNorm2D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        return self.norm(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W)
class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
# ConvNeXtBlock and ConvNeXtBlock1D are adapted from the ConvNeXt block (Liu et al., CVPR 2022), https://github.com/facebookresearch/ConvNeXt
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the MIT License.
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, k=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim)
        self.norm = LayerNorm2D(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                 requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)  # Apply norm in (N, C, H, W) format
        x = x.permute(0, 2, 3, 1)  # Back to (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, k=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=k, padding=k//2, groups=dim)
        self.norm = LayerNorm1D(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                 requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (N, H, C) -> (N, C, H)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, C, H) -> (N, H, C)
        x = input + self.drop_path(x)
        return x
# Cross-Attention Module for feature fusion
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        context_flat = context.flatten(2).transpose(1, 2)  # B, HW_ctx, C

        q = self.q(x_flat).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_flat = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_flat = self.proj(x_flat)
        x_flat = self.proj_drop(x_flat)
        
        return x_flat.transpose(1, 2).reshape(B, C, H, W)

class VGGEncoder(nn.Module):
    # Names correspond to the paper: conv1_1, conv2_1, conv3_1, conv4_1.
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  # type: ignore
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        # Layer indices for convX_1 when using torchvision's sequential list
        self.slice1 = vgg[:2]   # conv1_1 (ReLU after)
        self.slice2 = vgg[2:7]  # up to conv2_1
        self.slice3 = vgg[7:12]  # up to conv3_1
        self.slice4 = vgg[12:21]  # up to conv4_1
    def forward(self, x: torch.Tensor):
        #assume x is [B, 3, H, W] tensor in [0,1]
        x = (x - self.mean) / self.std
        out = {}
        x = self.slice1(x)
        out["conv1_1"] = x
        x = self.slice2(x)
        out["conv2_1"] = x
        x = self.slice3(x)
        out["conv3_1"] = x
        x = self.slice4(x)
        out["conv4_1"] = x
        return out

class LearnedSeparableGuide(nn.Module):
    def __init__(self, lut_size=(64, 64, 64)):
        super().__init__()
        self.lut_L = nn.Parameter(torch.zeros(lut_size[0], dtype=torch.float32))
        self.lut_Ct = nn.Parameter(torch.zeros(lut_size[1], dtype=torch.float32))
        self.lut_Cp = nn.Parameter(torch.zeros(lut_size[2], dtype=torch.float32))

    def _mono(self, w, lo, hi):
        # w: (>=n,) unconstrained → (n,) monotone in [lo,hi]
        s = F.softplus(w) + 1e-6                # positive
        c = torch.cumsum(s, dim=-1)
        c = c / c[-1].clamp_min(1e-6)           # [0,1]
        return lo + (hi - lo) * c               # [lo,hi], strictly increasing
    
    def batched_linear_interp_normalized(self, grid: torch.Tensor, x_norm: torch.Tensor):
        """
        Batched linear interpolation for 1D signals with normalized coordinates.

        grid: (B, C, L)
        x_norm: (B, C)  -- positions in [-1, 1]
        returns: (B, C)
        """
        L = grid.shape[-1]

        # Map [-1, 1] → [0, L-1]
        x = (x_norm + 1) * (L - 1) / 2
        x = x.clamp(0, L - 1 - 1e-6)

        x0 = x.floor().long()
        x1 = (x0 + 1).clamp(max=L - 1)
        w = x - x0.float()  # fractional weight

        g0 = torch.gather(grid, -1, x0.unsqueeze(-1)).squeeze(-1)
        g1 = torch.gather(grid, -1, x1.unsqueeze(-1)).squeeze(-1)

        return g0 * (1 - w) + g1 * w

    def forward(self, rgb):                         # rgb in [0,1], shape [B,3,H,W]
        lut_L = self._mono(self.lut_L, 0, 1)
        lut_Ct = self._mono(self.lut_Ct, 0.0, 1.0)
        lut_Cp = self._mono(self.lut_Cp, 0.0, 1.0)
        B, C, H, W = rgb.shape
        x_norm = torch.stack([rgb[:, 0], rgb[:, 1], rgb[:, 2]], dim=-1).reshape(B*H*W, 3) # [B,1,H,W,3]
        grid = torch.stack([lut_L, lut_Ct, lut_Cp], dim=0)[None].expand(B*H*W, -1, -1) # [B, 3, 64]
        return self.batched_linear_interp_normalized(grid, x_norm).reshape(B, H, W, 3).permute(0, 3, 1, 2)

class Grid(nn.Module):
    def __init__(self):
        super(Grid, self).__init__()
        self.img_projection = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2), # 16, 128, 128
            nn.GELU(),
            LayerNorm2D(16),
            ConvNeXtBlock(16),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), # 32, 64, 64
            nn.GELU(),
            LayerNorm2D(32),
            ConvNeXtBlock(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2), # 64, 32, 32
            nn.GELU(),
            LayerNorm2D(64),
            ConvNeXtBlock(64),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), # 128, 16, 16
            nn.GELU(),
            LayerNorm2D(128),
            ConvNeXtBlock(128), 
            nn.GELU(),
            LayerNorm2D(128),
            ConvNeXtBlock(128),
        )
        self.hist_projection = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=5, padding=2, stride=2),  # 16, 64, 64
            nn.GELU(),
            LayerNorm1D(16),
            ConvNeXtBlock1D(16),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2),  # 32, 32, 32
            nn.GELU(),
            LayerNorm1D(32),
            ConvNeXtBlock1D(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),  # 64, 16, 16
            nn.GELU(),
            LayerNorm1D(64),
            ConvNeXtBlock1D(64, k=5),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),  # 128, 8, 8
            nn.GELU(),
            LayerNorm1D(128),
            ConvNeXtBlock1D(128, k=3),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),  # 128, 4, 4
            nn.GELU(),
            LayerNorm1D(128),
            nn.Conv1d(128, 128, kernel_size=4), # 128, 1, 1
        )

        self.attention = CrossAttention(128, 8)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 130, kernel_size=3, padding=1),
            LayerNorm2D(130),
            nn.GELU(),
            ConvNeXtBlock(130),
            nn.Conv2d(130, 13*griddim, kernel_size=1),
        )

        self.fusion[-1].weight.data = self.fusion[-1].weight.data * 1e-2
        new_bias = torch.zeros_like(self.fusion[-1].bias.data).reshape(13, griddim)
        new_bias[12] = 1.0 #alpha is one
        new_bias[0] = 2.0 #L goes from [0, 1] to [-1, 1]
        new_bias[4] = 2.0 #Ct goes from [-0.5, 0.5] to [-1, 1]
        new_bias[8] = 2.0 #Cp goes from [-0.5, 0.5] to [-1, 1]
        new_bias[9:12] = -1.0 #bias since L goes from [0, 1] to [-1, 1]
        self.fusion[-1].bias.data = new_bias.view(-1)

        self.sampler = GridSampler()
        self.pre_guide = LearnedSeparableGuide()
        self.post_guide = LearnedSeparableGuide()

    def apply_grid(self, img_downed, hist):
        img_downed = self.img_projection(img_downed)
        hist = self.hist_projection(hist)
        img_downed = torch.cat([img_downed, self.attention(img_downed, hist)], dim=1)
        grid = self.fusion(img_downed).reshape(-1, 13, griddim, gridsize, gridsize)
        alpha = F.softplus(grid[:, -1:]) + 1e-3
        grid = torch.cat([grid[:, :-1] * alpha, alpha], dim=1)
        return grid

    def apply_coeffs(self, coeffs, img_content):
        B, _, H, W = img_content.shape
        coeffs, alpha = coeffs[:, :-1], coeffs[:, -1:]
        coeffs = coeffs / alpha
        coeffs = coeffs.flatten(2).permute(0, 2, 1)
        mats = coeffs[..., :9].reshape(B*H*W, 3, 3)
        bias = coeffs[..., 9:].reshape(B*H*W, 3, 1)
        im = self.pre_guide((img_content-0.5)*2.0).flatten(2).permute(0, 2, 1)[..., None].reshape(B*H*W, 3, 1)
        output = torch.bmm(mats, im) + bias
        output = self.post_guide(output.clamp(-1.0, 1.0).reshape(B, H, W, 3).permute(0, 3, 1, 2))
        return output

    def forward(self, img_content, hist_content, hist_style):
        hist = torch.cat([hist_content, hist_style], dim=1)
        if img_content.shape[2] != 64 or img_content.shape[3] != 64:
            resized = TF.resize(img_content, (64, 64), antialias=True)
            grid = self.apply_grid(resized, hist)
        else:
            grid = self.apply_grid(img_content, hist)
        coeffs = self.sampler(grid, img_content)
        lab = self.apply_coeffs(coeffs, img_content)
        return lab, grid

class Model(pl.LightningModule):
    def __init__(self, steps_per_epoch):
        super(Model, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.model = Grid().train()
        self.vgg = VGGEncoder()
        self.vgg.eval()
        self.vgg.requires_grad_(False)
    def forward(self, source, target):
        hist_source = batch_histogram_1D(source)
        hist_target = batch_histogram_1D(target)
        lab_hat, grid = self.model(source, hist_source, hist_target)
        return lab_hat, grid
    def vgg_loss(self, output, gt_img):
        gt_feat = self.vgg(gt_img)
        out_feat = self.vgg(output)
        spatial_loss = 0.0
        distribution_loss = 0.0
        for k in out_feat.keys():
            cl = (gt_feat[k] - out_feat[k]).pow(2).mean()
            sorted_gt_feat = torch.sort(gt_feat[k].flatten(-2), dim=-1)[0]
            sorted_out_feat = torch.sort(out_feat[k].flatten(-2), dim=-1)[0]
            dl = (sorted_gt_feat - sorted_out_feat).pow(2).mean()*2
            spatial_loss = spatial_loss + cl
            distribution_loss = distribution_loss + dl
        return spatial_loss, distribution_loss
        
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def training_step_compile(self, batch, batch_idx):
        source_srgb, gt_srgb = batch 

        source = srgb_to_ycbcr(source_srgb)
        gt = srgb_to_ycbcr(gt_srgb)

        hist_source = batch_histogram_1D(source)
        hist_gt = batch_histogram_1D(gt)

        lab_hat, grid = self.model(source_srgb + torch.randn_like(source_srgb) * 1e-4, hist_source + torch.randn_like(hist_source) * 1e-5, hist_gt + torch.randn_like(hist_gt) * 1e-5)

        spatial_loss, distribution_loss = self.vgg_loss(lab_hat, gt_srgb)
        total_loss = (spatial_loss + distribution_loss)/2
        return total_loss, lab_hat, grid, gt_srgb, source_srgb, spatial_loss, distribution_loss
    
    def training_step(self, batch, batch_idx):
        total_loss, lab_hat, grid, gt_srgb, source_srgb, spatial_loss, distribution_loss = self.training_step_compile(batch, batch_idx)
        
        if self.trainer.is_global_zero:
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_spatial_loss", spatial_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_distribution_loss", distribution_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def validation_step_compile(self, batch, batch_idx):
        source_srgb, gt_srgb = batch 

        source = srgb_to_ycbcr(source_srgb)
        gt = srgb_to_ycbcr(gt_srgb)

        hist_source = batch_histogram_1D(source)
        hist_gt = batch_histogram_1D(gt)
        

        lab_hat, grid = self.model(source_srgb, hist_source, hist_gt)

        spatial_loss, distribution_loss = self.vgg_loss(lab_hat, gt_srgb)
        total_loss = (spatial_loss + distribution_loss)/2
        return total_loss, lab_hat, grid, gt_srgb, source_srgb, spatial_loss, distribution_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        total_loss, lab_hat, grid, gt_srgb, source_srgb, spatial_loss, distribution_loss = self.validation_step_compile(batch, batch_idx)

        if self.trainer.is_global_zero: 
            self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spatial_loss", spatial_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_distribution_loss", distribution_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self): #didnt play with this much.
        base_lr = 3e-4                  # cap for stability

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,  # Always start and target base_lr after warmup
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,  # start very low (almost zero)
                end_factor=1.0,
                total_iters=self.steps_per_epoch,  # warmup for single epoch
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def verify_srgb(folder):
    """One-time check that every image under folder is sRGB (no embedded profile = sRGB); remembered in <folder>/.srgb_verified."""
    marker = os.path.join(folder, ".srgb_verified")
    if os.path.exists(marker):
        return
    from PIL import ImageCms
    from concurrent.futures import ThreadPoolExecutor
    files = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
             if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._") and "__MACOSX" not in r]  # skip macOS resource forks
    def check(f):
        with Image.open(f) as im:
            icc = im.info.get("icc_profile")
        if not icc:
            return None
        try:
            prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            ident = (ImageCms.getProfileName(prof) + " " + ImageCms.getProfileDescription(prof)).lower()
        except Exception as e:
            return f"{f}: unreadable ICC profile ({e})"
        return None if "srgb" in ident else f"{f}: embedded profile is not sRGB ({' '.join(ident.split())})"
    print(f"checking colour profiles of {len(files)} images in {folder} (once)")
    with ThreadPoolExecutor(16) as pool:
        bad = [b for b in pool.map(check, files) if b]
    if bad:
        raise RuntimeError(f"{len(bad)} image(s) in {folder} are not sRGB; convert them to sRGB first (no embedded profile = sRGB), e.g.\n  " + "\n  ".join(bad[:10]))
    open(marker, "w").write("all images sRGB (no embedded profile = sRGB)\n")


def main():
    for name, url in (("flux_style_ds_srgb", flux_style_ds_srgb_tar), ("unsplash25k_512_srgb", unsplash25k_512_srgb_tar)):  # EvaluationSet is fetched by eval.py
        download_and_extract(name, url)
        verify_srgb(os.path.join(BASE_DIR, "data", name))

    torch.cuda.set_device("cuda:" + device[0])
    pl.seed_everything(args.seed, workers=True)
    train_loader, val_loader = make_dataloaders()

    model = Model(steps_per_epoch=len(train_loader))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=device,
        max_epochs=args.max_epochs+1,
        logger=False,  # no experiment logger: the progress bar shows the losses, the checkpoints are the output
        enable_progress_bar=True,
        enable_model_summary=True,
        precision=dtype,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=-1 if args.save_all else 1,
                **({"monitor": "val_loss", "mode": "min"} if args.save_all else {}),
                every_n_epochs=1,
            ),
        ],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

