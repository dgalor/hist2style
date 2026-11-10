from model import Model, batch_histogram_1D, srgb_to_ycbcr
import os
import sys
import torch
import time
import warnings
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F

checkpoint_forced = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-epoch=1500-val_loss=2.79.ckpt")  # next to this file, independent of cwd


def get_device() -> str:
	if torch.cuda.is_available():
		return "cuda:0"
	if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
		return "mps"

	cuda_built = torch.version.cuda is not None
	if sys.platform in ("linux", "win32") and not cuda_built:
		warnings.warn(
			"This PyTorch install is CPU-only (no CUDA wheel). "
			"Reinstall torch/torchvision from PyPI without a cpu index URL. "
			"Falling back to CPU; this will be slow.",
			stacklevel=2,
		)
	elif cuda_built:
		warnings.warn(
			f"PyTorch was built with CUDA {torch.version.cuda}, but no GPU is usable "
			"(missing NVIDIA device, driver too old for CUDA 13.0, or runtime mismatch). "
			"Falling back to CPU; this will be slow.",
			stacklevel=2,
		)
	else:
		warnings.warn("No CUDA or MPS device found; falling back to CPU. This will be slow.", stacklevel=2)
	return "cpu"


DEVICE = get_device()


def forward(model, source_srgb, target_hist, grid_op=lambda x: x, lut_op=lambda x: x):
	source = srgb_to_ycbcr(source_srgb)
	hist_source = batch_histogram_1D(source)
	lab_hat, _ = model.model(source_srgb, hist_source, target_hist, grid_op, lut_op)
	return lab_hat


def load_model():
	print(checkpoint_forced)
	loaded = Model.load_from_checkpoint(checkpoint_forced, map_location=DEVICE, strict=False)
	loaded.eval()
	loaded.to(device=DEVICE, dtype=torch.float32)
	return loaded


model = load_model()


def np_hist_1D(img: np.ndarray, num_bins: int, density: bool = True, icpct: bool = True) -> np.ndarray:
	"""
	img: [N, C] float in [0,1]
	returns: [C, num_bins] histogram
	"""
	_, C = img.shape
	hist = np.zeros((C, num_bins), dtype=np.float32)
	if icpct:
		assert C == 3, "ICtCp has 3 channels"
		img[:, 1:] = img[:, 1:] + 0.5
	for c in range(C):
		coords = np.floor(img[:, c] * num_bins).astype(np.int64)
		np.clip(coords, 0, num_bins - 1, out=coords)
		np.add.at(hist[c], coords.ravel(), 1)
	if density:
		hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1e-8)
	return hist


def downsample_tensor(img: torch.Tensor, max_pixels: int = 1_000_000) -> torch.Tensor:
	"""Downsample the image to max_pixels while preserving aspect ratio. img is [B, C, H, W]."""
	w0 = img.shape[3]
	h0 = img.shape[2]
	px = w0 * h0
	if px > max_pixels:
		s = (float(max_pixels) / float(px)) ** 0.5
		w = int(round(w0 * s))
		h = int(round(h0 * s))
		img = F.interpolate(img, (h, w), mode="bilinear", align_corners=False, antialias=True)
	return img


@torch.no_grad()
def invert_3x3_field(M):
	"""
	Invert per-pixel 3x3 matrices.
	Input: M of shape (B, 9, H, W)
	Output: invM of shape (B, 9, H, W)
	"""
	B, _, H, W = M.shape
	M = M.view(B, 3, 3, H, W)

	det = (
		M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1]) -
		M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) +
		M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0])
	)

	inv_det = 1.0 / (det + 1e-8)

	adj = torch.empty_like(M)
	adj[:, 0, 0] = M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1]
	adj[:, 0, 1] = M[:, 0, 2] * M[:, 2, 1] - M[:, 0, 1] * M[:, 2, 2]
	adj[:, 0, 2] = M[:, 0, 1] * M[:, 1, 2] - M[:, 0, 2] * M[:, 1, 1]
	adj[:, 1, 0] = M[:, 1, 2] * M[:, 2, 0] - M[:, 1, 0] * M[:, 2, 2]
	adj[:, 1, 1] = M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0]
	adj[:, 1, 2] = M[:, 0, 2] * M[:, 1, 0] - M[:, 0, 0] * M[:, 1, 2]
	adj[:, 2, 0] = M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0]
	adj[:, 2, 1] = M[:, 0, 1] * M[:, 2, 0] - M[:, 0, 0] * M[:, 2, 1]
	adj[:, 2, 2] = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]

	invM = adj * inv_det.unsqueeze(1).unsqueeze(1)
	return invM.view(B, 9, H, W)


@torch.no_grad()
def apply_model(img, r, g, b, max_pixels=1000000, amount=1.0, num_iters=1):
	t = time.time()
	target_hist = torch.from_numpy(np.stack([r, g, b], axis=0)).to(DEVICE)
	assert target_hist.shape == (3, 128)
	summed = target_hist.sum(dim=1)
	assert summed.allclose(torch.ones_like(summed))
	print(f"Time to convert target hist: {time.time() - t} seconds")
	t = time.time()
	img = torch.from_numpy(img).to(DEVICE).permute(2, 0, 1)[None]
	print(f"time to convert img to torch {DEVICE}", time.time() - t)
	t = time.time()

	assert img.dtype == torch.uint8
	img = img.float() / 255.0
	print(f"Time to convert img to float: {time.time() - t} seconds")
	t = time.time()
	img_torch = downsample_tensor(img, max_pixels=max_pixels)

	def grid_op(grid):
		identity = torch.tensor(
			[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, -1.0, -1.0, -1.0],
			device=grid.device,
			dtype=grid.dtype,
		).reshape(1, 12, 1, 1)
		if amount < 0.0:
			new_grid = grid.clone()
			new_grid[:, :9] = invert_3x3_field(grid[:, :9])
			new_grid[:, 9:12] *= -1.0
			grid[:, :12] = (1 + amount) * identity + (-amount) * new_grid[:, :12]
		else:
			grid[:, :12] = (1 - amount) * identity + amount * grid[:, :12]
		return grid

	def lut_op(lut):
		a = min(1.0, abs(amount))
		return lut * a + (1 - a) * torch.linspace(lut[0], lut[-1], lut.shape[0], device=lut.device, dtype=lut.dtype)

	out_img = img_torch
	for _ in range(num_iters):
		out_img = forward(model, out_img, target_hist[None], grid_op, lut_op)

	print(f"Time to apply model: {time.time() - t} seconds")
	t = time.time()
	out_img = out_img.cpu().float()[0]
	print(f"Time to convert out img: {time.time() - t} seconds")
	t = time.time()

	out_img = TF.to_pil_image(out_img)
	print(f"Time to convert out img jpeg to pil: {time.time() - t} seconds")
	return out_img
