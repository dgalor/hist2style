import io
import base64
import json
import sys
import logging
import time
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
from PIL import Image, ImageCms, ImageOps
import numpy as np
import torch
from starlette.responses import Response

torch.set_float32_matmul_precision("medium")

HIGHLIGHT_FACTOR = 1.0

_DEMO_DIR = Path(__file__).resolve().parent / "demo"
if str(_DEMO_DIR) not in sys.path:
	sys.path.insert(0, str(_DEMO_DIR))

from utils import apply_model, np_hist_1D, srgb_to_ycbcr, DEVICE  # demo/utils.py (loads the checkpoint next to it)

app = FastAPI()

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("timing")


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
	start = time.perf_counter()
	try:
		response: Response = await call_next(request)
		status = response.status_code
	except Exception:
		duration_ms = (time.perf_counter() - start) * 1000.0
		logger.exception(
			"REQ %s %s -> EXC after %.2f ms",
			request.method, request.url.path, duration_ms,
		)
		raise
	else:
		duration_ms = (time.perf_counter() - start) * 1000.0
		response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
		logger.info(
			"REQ %s %s -> %d in %.2f ms | UA=%s",
			request.method,
			request.url.path,
			status,
			duration_ms,
			request.headers.get("user-agent", "-")[:80],
		)
		return response

static_dir = _DEMO_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
	with open(static_dir / "index.html", "r", encoding="utf-8") as f:
		html = f.read()
	return HTMLResponse(content=html)


_srgb_profile = ImageCms.createProfile("sRGB")


def open_image(src) -> Image.Image:
	"""Open an image upright: apply its EXIF orientation (phone photos), as run.py does."""
	return ImageOps.exif_transpose(Image.open(src))


def to_srgb(img: Image.Image) -> Image.Image:
	"""Inputs must be sRGB: no embedded profile is treated as sRGB, an sRGB profile passes, anything else is rejected."""
	icc_profile_source = img.info.get("icc_profile")
	if not icc_profile_source:
		return img
	try:
		source_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_source))
		identity = " ".join((ImageCms.getProfileName(source_prof), ImageCms.getProfileDescription(source_prof))).lower()
	except Exception as error:
		raise ValueError(f"unreadable ICC profile ({error}); convert the image to sRGB first") from error
	if "srgb" not in identity:
		raise ValueError(f"image is not sRGB (embedded profile: {' '.join(identity.split()) or 'unknown'}); convert it to sRGB first")
	return img


class HistURLIn(BaseModel):
	url: str
	num_bins: int | None = None
	density: bool = True


def _pil_to_float_rgb(arr: np.ndarray, subsample=100000) -> np.ndarray:
	"""Return float32 [N,3] in [0,1], preserving 16-bit when present."""
	assert arr.shape[2] == 3, "Image must have 3 channels"

	dtype = arr.dtype
	if dtype == np.uint16:
		print("HDR reference detected")
		arr = (arr.astype(np.float32) / 65535.0).clip(0, 1)
	elif dtype == np.uint8:
		arr = (arr.astype(np.float32) / 255.0 * HIGHLIGHT_FACTOR).clip(0, 1)
	else:
		raise ValueError(f"Unsupported dtype: {dtype}")
	arr = srgb_to_ycbcr(torch.from_numpy(arr).to(DEVICE).permute(2, 0, 1)).permute(1, 2, 0).cpu().numpy()
	arr = arr.reshape(-1, 3)
	if arr.shape[0] > subsample:
		arr = arr[np.random.choice(arr.shape[0], subsample, replace=False)]
	return arr


async def _load_image_from_multipart(upload: UploadFile) -> np.ndarray:
	data = await upload.read()
	name = (upload.filename or "").lower()
	if name.endswith((".avif", ".heif", ".heic")):
		raise ValueError("AVIF/HEIF not supported for now")
	return np.array(to_srgb(open_image(io.BytesIO(data)).convert("RGB")))


async def _load_image_from_url(url: str) -> np.ndarray:
	if "/static/" in url:
		filename = url.split("/static/")[-1]
		static_file_path = static_dir / filename
		if static_file_path.exists():
			return np.array(to_srgb(open_image(static_file_path).convert("RGB")))

	async with httpx.AsyncClient(timeout=20) as client:
		r = await client.get(url)
		r.raise_for_status()
		content = r.content
	low = url.lower()
	if low.endswith((".avif", ".heif", ".heic")):
		raise ValueError("AVIF/HEIF not supported for now")
	return np.array(to_srgb(open_image(io.BytesIO(content)).convert("RGB")))


@app.post("/hists")
async def histograms(
	request: Request,
	file: UploadFile | None = File(None),
	num_bins: int | None = Form(None),
	density: bool = Form(True),
):
	"""
	Accepts:
		- multipart/form-data: file, optional num_bins, density
		- application/json: { "url": "...", "num_bins": 65536, "density": true }
	Returns:
		{ "hist_r": [...], "hist_g": [...], "hist_b": [...] }
	"""
	bins = num_bins or 128

	t = time.time()
	if request.headers.get("content-type", "").startswith("application/json"):
		body = await request.json()
		inp = HistURLIn(**body)
		bins = inp.num_bins or bins
		density = bool(inp.density)
		img_pil = await _load_image_from_url(inp.url)
	else:
		if file is None:
			img_pil = np.array(to_srgb(open_image(static_dir / "yellowFlower.jpg").convert("RGB")))
		else:
			img_pil = await _load_image_from_multipart(file)
	print(f"Image load time: {time.time() - t} seconds")
	t = time.time()
	if isinstance(img_pil, np.ndarray):
		img_pil_pil = Image.fromarray(img_pil)
	else:
		img_pil_pil = img_pil
	img_pil_pil = img_pil_pil.resize((256, 256), Image.Resampling.LANCZOS)
	img_pil = np.array(img_pil_pil)
	img_f = _pil_to_float_rgb(img_pil)
	print(f"Image to float time: {time.time() - t} seconds")
	t = time.time()

	H = np_hist_1D(img_f, num_bins=bins, density=density)
	print(f"Hist compute time: {time.time() - t} seconds")
	t = time.time()
	result = {
		"hist_r": H[0].astype(float).tolist(),
		"hist_g": H[1].astype(float).tolist(),
		"hist_b": H[2].astype(float).tolist(),
	}
	print(f"Result time: {time.time() - t} seconds")
	return JSONResponse(result)


@app.post("/apply")
async def apply(
	file: Optional[UploadFile] = File(None),
	hist_r: Optional[str] = Form(None),
	hist_g: Optional[str] = Form(None),
	hist_b: Optional[str] = Form(None),
	max_pixels: int = Form(1000000),
	quality: int = Form(60),
	amount: float = Form(1.0),
	num_iters: int = Form(1),
):
	print(f"Max pixels: {max_pixels}")
	start_time = time.time()
	if file is not None:
		img_bytes = await file.read()
		print(file.filename)
		if file.filename and file.filename.endswith(".avif"):
			raise ValueError("AVIF/HEIF not supported for now")
		image = np.array(to_srgb(open_image(io.BytesIO(img_bytes)).convert("RGB")))
	else:
		image = np.array(to_srgb(open_image(static_dir / "yellowFlower.jpg").convert("RGB")))
	print(f"File read time: {time.time() - start_time} seconds")
	start_time = time.time()

	def _parse_hist(s: Optional[str]) -> Optional[np.ndarray]:
		if s is None:
			return None
		try:
			arr = np.array(json.loads(s), dtype=np.float64).reshape(128)
			if not np.isfinite(arr).all():
				return None
			arr = np.maximum(arr, 0)
			if arr.sum() <= 0:
				return None
			return (arr / arr.sum()).astype(np.float32)
		except Exception:
			return None

	max_pixels = min(max(int(max_pixels), 50_000), 10_000_000)  # the UI's MaxPx range
	num_iters = min(max(int(num_iters), 1), 10)  # the UI's NumIters range
	target_hists = {
		"r": _parse_hist(hist_r),
		"g": _parse_hist(hist_g),
		"b": _parse_hist(hist_b),
	}
	print(f"Hist parse time: {time.time() - start_time} seconds")
	start_time = time.time()
	image = apply_model(
		image,
		target_hists["r"],
		target_hists["g"],
		target_hists["b"],
		max_pixels=max_pixels,
		amount=amount,
		num_iters=num_iters,
	)
	print(f"Apply model time jpeg: {time.time() - start_time} seconds")
	start_time = time.time()
	buf = io.BytesIO()
	image.save(buf, format="JPEG", quality=quality)
	b64 = base64.b64encode(buf.getvalue()).decode("ascii")
	print(f"Base64 encode time: {time.time() - start_time} seconds")
	return JSONResponse({
		"imageBase64": f"data:image/jpeg;base64,{b64}",
	})


if __name__ == "__main__":
	import uvicorn
	from fastapi.middleware.cors import CORSMiddleware

	app.add_middleware(
		CORSMiddleware,
		allow_origins=[
			"http://localhost",
			"http://localhost:8000",
		],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	uvicorn.run("demo:app", host="0.0.0.0", port=8000)
