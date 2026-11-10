import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import lightning as pl

gridsize = 16
histogram_size = 128
griddim = 8

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
	mat = _RGB2YCBCR.to(dtype=img.dtype, device=img.device)
	out = img.movedim(-3, -1) @ mat.T
	out = torch.cat([out[..., 0:1].clip(0.0, 1.0), out[..., 1:3].clip(-0.5, 0.5)], dim=-1)
	return out.movedim(-1, -3)


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

		yy = (2 * torch.arange(H, device=img_content.device).float() + 1) / H - 1
		xx = (2 * torch.arange(W, device=img_content.device).float() + 1) / W - 1
		Y, X = torch.meshgrid(yy, xx, indexing="ij")
		x, y = X.view(1, H, W).expand(B, -1, -1), Y.view(1, H, W).expand(B, -1, -1)
		w = F.softmax(self.L_weights, dim=1)
		pre = self.pre_guide(img_content)
		z = (pre * w).sum(1)
		grid3d = torch.stack([x, y, (z - 0.5) * 2], dim=-1).view(B, 1, H, W, 3).clip(-1, 1)
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
		self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
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
		x = self.norm(x)
		x = x.permute(0, 2, 3, 1)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		if self.gamma is not None:
			x = self.gamma * x
		x = x.permute(0, 3, 1, 2)
		x = input + self.drop_path(x)
		return x


class ConvNeXtBlock1D(nn.Module):
	def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, k=7):
		super().__init__()
		self.dwconv = nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
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
		x = x.permute(0, 2, 1)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		if self.gamma is not None:
			x = self.gamma * x
		x = x.permute(0, 2, 1)
		x = input + self.drop_path(x)
		return x


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
		x_flat = x.flatten(2).transpose(1, 2)
		context_flat = context.flatten(2).transpose(1, 2)

		q = self.q(x_flat).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		kv = self.kv(context_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		k, v = kv[0], kv[1]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x_flat = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
		x_flat = self.proj(x_flat)
		x_flat = self.proj_drop(x_flat)

		return x_flat.transpose(1, 2).reshape(B, C, H, W)


class LearnedSeparableGuide(nn.Module):
	def __init__(self, lut_size=(64, 64, 64)):
		super().__init__()
		self.lut_L = nn.Parameter(torch.zeros(lut_size[0], dtype=torch.float32))
		self.lut_Ct = nn.Parameter(torch.zeros(lut_size[1], dtype=torch.float32))
		self.lut_Cp = nn.Parameter(torch.zeros(lut_size[2], dtype=torch.float32))

	def _mono(self, w, lo, hi):
		s = F.softplus(w) + 1e-6
		c = torch.cumsum(s, dim=-1)
		c = c / c[-1].clamp_min(1e-6)
		return lo + (hi - lo) * c

	def batched_linear_interp_normalized(self, grid: torch.Tensor, x_norm: torch.Tensor):
		"""
		Batched linear interpolation for 1D signals with normalized coordinates.

		grid: (B, C, L)
		x_norm: (B, C)  -- positions in [-1, 1]
		returns: (B, C)
		"""
		L = grid.shape[-1]

		x = (x_norm + 1) * (L - 1) / 2
		x = x.clamp(0, L - 1 - 1e-6)

		x0 = x.floor().long()
		x1 = (x0 + 1).clamp(max=L - 1)
		w = x - x0.float()

		g0 = torch.gather(grid, -1, x0.unsqueeze(-1)).squeeze(-1)
		g1 = torch.gather(grid, -1, x1.unsqueeze(-1)).squeeze(-1)

		return g0 * (1 - w) + g1 * w

	def forward(self, rgb, lut_op=lambda x: x):
		lut_L = lut_op(self._mono(self.lut_L, 0, 1))
		lut_Ct = lut_op(self._mono(self.lut_Ct, 0.0, 1.0))
		lut_Cp = lut_op(self._mono(self.lut_Cp, 0.0, 1.0))
		B, C, H, W = rgb.shape
		x_norm = torch.stack([rgb[:, 0], rgb[:, 1], rgb[:, 2]], dim=-1).reshape(B * H * W, 3)
		grid = torch.stack([lut_L, lut_Ct, lut_Cp], dim=0)[None].expand(B * H * W, -1, -1)
		return self.batched_linear_interp_normalized(grid, x_norm).reshape(B, H, W, 3).permute(0, 3, 1, 2)


def lowpass(x, size=5, sigma=None):
	def make_gaussian_kernel(size=11, sigma=None, device=None, dtype=torch.float32):
		if sigma is None:
			sigma = size / 6
		ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
		xx, yy = torch.meshgrid(ax, ax, indexing="xy")
		kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
		kernel = kernel / kernel.sum()
		return kernel[None, None]

	B = x.shape[0]
	x = x.reshape(-1, 1, gridsize, gridsize)
	device = x.device
	dtype = x.dtype
	kernel = make_gaussian_kernel(size=size, sigma=sigma, device=device, dtype=dtype)
	pad = size // 2
	x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
	x = F.conv2d(x, kernel.expand(x.size(1), 1, size, size), groups=x.size(1))
	return x.reshape(B, -1, griddim, gridsize, gridsize)


class Grid(nn.Module):
	def __init__(self):
		super(Grid, self).__init__()
		self.img_projection = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2),
			nn.GELU(),
			LayerNorm2D(16),
			ConvNeXtBlock(16),
			nn.Conv2d(16, 32, kernel_size=5, padding=2),
			nn.GELU(),
			LayerNorm2D(32),
			ConvNeXtBlock(32),
			nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),
			nn.GELU(),
			LayerNorm2D(64),
			ConvNeXtBlock(64),
			nn.Conv2d(64, 128, kernel_size=5, padding=2),
			nn.GELU(),
			LayerNorm2D(128),
			ConvNeXtBlock(128),
			nn.GELU(),
			LayerNorm2D(128),
			ConvNeXtBlock(128),
		)
		self.hist_projection = nn.Sequential(
			nn.Conv1d(6, 16, kernel_size=5, padding=2, stride=2),
			nn.GELU(),
			LayerNorm1D(16),
			ConvNeXtBlock1D(16),
			nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2),
			nn.GELU(),
			LayerNorm1D(32),
			ConvNeXtBlock1D(32),
			nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
			nn.GELU(),
			LayerNorm1D(64),
			ConvNeXtBlock1D(64, k=5),
			nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
			nn.GELU(),
			LayerNorm1D(128),
			ConvNeXtBlock1D(128, k=3),
			nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),
			nn.GELU(),
			LayerNorm1D(128),
			nn.Conv1d(128, 128, kernel_size=4),
		)

		self.attention = CrossAttention(128, 8)
		self.fusion = nn.Sequential(
			nn.Conv2d(256, 130, kernel_size=3, padding=1),
			LayerNorm2D(130),
			nn.GELU(),
			ConvNeXtBlock(130),
			nn.Conv2d(130, 13 * griddim, kernel_size=1),
		)

		self.fusion[-1].weight.data = self.fusion[-1].weight.data * 1e-2
		new_bias = torch.zeros_like(self.fusion[-1].bias.data).reshape(13, griddim)
		new_bias[12] = 1.0
		new_bias[0] = 2.0
		new_bias[4] = 2.0
		new_bias[8] = 2.0
		new_bias[9:12] = -1.0
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

	def apply_coeffs(self, coeffs, img_content, lut_op=lambda x: x, grid_op=lambda x: x):
		B, _, H, W = img_content.shape
		coeffs, alpha = coeffs[:, :-1], coeffs[:, -1:]
		coeffs = coeffs / alpha
		coeffs = grid_op(coeffs)
		coeffs = coeffs.flatten(2).permute(0, 2, 1)
		mats = coeffs[..., :9].reshape(B * H * W, 3, 3)
		bias = coeffs[..., 9:].reshape(B * H * W, 3, 1)
		im = self.pre_guide((img_content - 0.5) * 2.0, lut_op).flatten(2).permute(0, 2, 1)[..., None].reshape(B * H * W, 3, 1)
		output = torch.bmm(mats, im) + bias
		output = self.post_guide(output.clamp(-1.0, 1.0).reshape(B, H, W, 3).permute(0, 3, 1, 2), lut_op)
		return output

	def forward(self, img_content, hist_content, hist_style, grid_op=lambda x: x, lut_op=lambda x: x):
		hist = torch.cat([hist_content, hist_style], dim=1)
		if img_content.shape[2] != 64 or img_content.shape[3] != 64:
			resized = TF.resize(img_content, (64, 64), antialias=True)
			grid = self.apply_grid(resized, hist)
		else:
			grid = self.apply_grid(img_content, hist)
		grid = lowpass(grid)
		coeffs = self.sampler(grid, img_content)
		lab = self.apply_coeffs(coeffs, img_content, lut_op, grid_op)
		return lab, grid


class Model(pl.LightningModule):
	def __init__(self, lowpass=False):
		super(Model, self).__init__()
		assert not lowpass, "lowpass is not supported for this model"
		self.model = Grid()

	def forward(self, source_srgb, target_srgb, grid_op=lambda x: x, lut_op=lambda x: x):
		source = srgb_to_ycbcr(source_srgb)
		target = srgb_to_ycbcr(target_srgb)
		hist_source = batch_histogram_1D(source)
		hist_target = batch_histogram_1D(target)
		lab_hat, grid = self.model(source_srgb, hist_source, hist_target, grid_op, lut_op)
		return lab_hat
