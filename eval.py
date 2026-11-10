#!/usr/bin/env python3
"""Benchmark a Hist2Style checkpoint on EvaluationSet (136 contents x 19 styles = 2584 content/style pairs).

    python3 eval.py                      # the released checkpoint demo/model-epoch=1500-val_loss=2.79.ckpt, all pairs, auto-picked idle GPU
    python3 eval.py --checkpoints my.ckpt other.ckpt
    python3 eval.py --epochs 1500 --experiment Hist2Style      # ./checkpoints/Hist2Style/model-epoch=1500-*.ckpt (train.py output)
    python3 eval.py --limit 100 --seed 0                        # quick check on a seeded subset (FID is then not comparable)

The table printed at the end contains a REFERENCE row with the numbers reported in README.md for demo/model-epoch=1500-val_loss=2.79.ckpt; a default run on
that checkpoint should reproduce them (up to small GPU / library differences, see README.md).

Metrics (lower is better for all):
  Y' / Cb / Cr  colour matching: stylized output and the STYLE image are both PIL-LANCZOS resized to 256x256, converted to Y'CbCr, and per
                channel the 65536 sorted values are compared by MSE (a 1-D quantile distance) x 255^2.
  Cycle         cycle consistency: cycle = model(stylized, content256), where stylized = the DECODED stylized JPEG (q95 4:2:0, exactly the
                saved file) and content256 = the full-res content resized to 256x256 (bilinear + antialias, the same path the style image
                takes), run by the checkpoint under evaluation; cycle and the ORIGINAL full-res content are both PIL-LANCZOS resized to
                1500x1500; RGB MSE x 255^2.
  FID           per-channel float32 PIL bicubic resize to 299x299, (x-128)/128, pytorch-fid InceptionV3 (pt_inception-2015-12-05-6726825d.pth,
                2048-d pool features), float64 Frechet distance; generated = all stylized outputs, real = the unique real images
                (136 contents + 19 styles); compare FID values within one table.
Pipeline per pair: content -> float, bilinear+antialias downsample to <= --max-pixels (size truncated); style -> 256x256 bilinear+antialias;
out = Model(content, style256); cycle = Model(decode(jpeg(out)), content256). Outputs are rounded to uint8 and encoded as JPEG q95 4:2:0;
the PRIMARY metrics are measured from those decoded JPEG bytes (= the files written with --image-format jpg); the same metrics measured from
the lossless uint8 outputs are stored alongside (*_lossless columns). Reductions are exact (int64) or float64. fp32 model; cudnn.benchmark off.
Input images must be sRGB: an embedded non-sRGB ICC profile is an error (--ignore-icc overrides); no profile is treated as sRGB.

Layout: <base-dir>/eval_table/<cfg>/<ckpt-stem>__<sha256[:10]>/{stylized/, cycle/, metrics.csv, inception_feats*.npy, summary.json, done.json}
  <cfg> = (all | n{limit}-seed{seed})-mp{max_pixels}-ds{dataset id}[-tf32][-nolp]; real-image inception features are cached in <cfg>/_real/.
A rerun skips finished checkpoints (done.json), resumes unfinished ones pair by pair and prints the table (table.md / table.csv / summary.png,
heatmap_*.png, examples_*.png next to it). EvaluationSet is downloaded to --data-dir on first use (same location as run.py).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# --- GPU selection must happen before torch is imported (torch device index != nvidia-smi index).
def pick_gpu(spec: str) -> int | None:
    if spec == "env":
        return None
    if spec != "auto":
        return int(spec)
    if "CUDA_VISIBLE_DEVICES" in os.environ:  # already chosen, e.g. by a job scheduler
        print(f"GPU: keeping CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        return None
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):  # no NVIDIA driver (e.g. macOS or CPU only): torch picks the device
        return None
    busy = {line.strip() for line in apps.splitlines() if line.strip()}
    best = None
    for line in gpus.splitlines():
        idx, uuid, used, total = [x.strip() for x in line.split(",")]
        if uuid in busy:
            continue
        free = int(total) - int(used)
        if best is None or free > best[1]:
            best = (int(idx), free)
    if best is None:
        raise SystemExit("No idle GPU (every GPU has a running process); pass --gpu N explicitly.")
    print(f"GPU auto-select: nvidia-smi index {best[0]} ({best[1]} MiB free, no running process)")
    return best[0]


if __name__ == "__main__":  # DataLoader workers re-import this file (as __mp_main__) and must not pick a GPU again
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--gpu", default="auto")
    _gpu = pick_gpu(_pre.parse_known_args()[0].gpu)
    if _gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # scipy sqrtm would otherwise grab every core
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import PIL  # noqa: E402
from PIL import Image, ImageCms  # noqa: E402
from scipy import linalg  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run  # noqa: E402  (data handling shared with run.py)
import model as model_module  # noqa: E402
from model import Model  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
DEFAULT_BASE_DIR = ROOT  # outputs in ./eval_table/, train.py checkpoints in ./checkpoints/<experiment>/
CKPT_RE = re.compile(r"model-epoch=(\d+)-val_loss=([\d.]+)\.ckpt$")
PAIR_RE = re.compile(r"(c-c\d+)_(s-s\d+)$")
METRIC_VERSION = 3  # bump when a metric definition changes -> finished checkpoints are re-evaluated
INCEPTION_NAME = "pt_inception-2015-12-05-6726825d.pth"
INCEPTION_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/" + INCEPTION_NAME
# numbers reported in README.md for the released checkpoint (python3 eval.py with defaults); printed/plotted next to every result for easy comparison
REFERENCE_ROW = {"label": "reference: demo/model-epoch=1500-val_loss=2.79.ckpt as reported in README.md", "Y": 322.49, "Cb": 49.21, "Cr": 81.10, "Cycle": 361.98, "FID": 70.21}
SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
RGB2YCBCR = np.array([[0.2126, 0.7152, 0.0722], [-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]], dtype=np.float64)  # == model.py
COLS = ("Y", "Cb", "Cr", "Cycle", "FID")
C_BLUE, C_ORANGE, C_TEXT, C_TEXT2, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="*", help="Checkpoint files (default: demo/model-epoch=1500-val_loss=2.79.ckpt when --epochs is not given).")
    p.add_argument("--epochs", type=int, nargs="*", default=[], help="Epochs resolved in <base-dir>/checkpoints/<experiment>/ (newest file if several).")
    p.add_argument("--experiment", default="Hist2Style", help="Sub-folder of <base-dir>/checkpoints/ used by --epochs.")
    p.add_argument("--checkpoint-dir", type=Path, help="Override the checkpoint directory used by --epochs.")
    p.add_argument("--allow-fresh", action="store_true", help="Evaluate a checkpoint modified <2 min ago (a training run may still be writing it).")
    p.add_argument("--gpu", default="auto", help="nvidia-smi GPU index, 'auto' (CUDA_VISIBLE_DEVICES if set, else the idle GPU with the most free memory) or 'env' (keep CUDA_VISIBLE_DEVICES).")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Large storage root; outputs go to <base-dir>/eval_table/.")
    # same data flags as run.py
    p.add_argument("--data-dir", type=Path, default=run.DEFAULT_DATA_DIR)
    p.add_argument("--evaluation-dir", type=Path)
    p.add_argument("--evaluation-url", default=run.DEFAULT_EVAL_URL)
    p.add_argument("--content-dir", type=Path)
    p.add_argument("--style-dir", type=Path)
    p.add_argument("--ignore-icc", action="store_true", help="Skip the sRGB-profile check and feed raw pixel values.")
    p.add_argument("--max-pixels", type=int, default=2_000_000, help="Content is downsampled to at most this many pixels before the model.")
    p.add_argument("--limit", type=int, default=0, help="Seeded random sample of this many pairs (0 = all pairs). FID is then on the subset only (marked *).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true", help="Allow TF32 matmul/conv on NVIDIA GPUs (faster, slightly different numbers); default off = fp32.")
    p.add_argument("--no-lowpass", action="store_true", help="Disable the inference-only 5x5 Gaussian low-pass of the bilateral grid (demo/model.py) in both passes.")
    # output control
    p.add_argument("--image-format", choices=["jpg", "png"], default="jpg", help="jpg = the measured q95 4:2:0 bytes; png = lossless uint8 outputs.")
    p.add_argument("--no-images", action="store_true", help="Compute metrics only; do not write stylized/cycle images.")
    p.add_argument("--workers", type=int, default=12, help="JPEG-decode worker processes and post-processing threads.")
    p.add_argument("--overwrite", action="store_true", help="Discard cached results of the requested checkpoints and recompute.")
    p.add_argument("--table-all", action="store_true", help="Also list every other finished checkpoint under this config in the table.")
    p.add_argument("--examples", type=int, default=3, help="Pairs shown in the example figure per checkpoint (0 = none).")
    return p.parse_args()


# ----------------------------------------------------------------------------- checkpoints
def resolve_checkpoints(args: argparse.Namespace) -> list[dict]:
    paths = list(args.checkpoints or [])
    if not paths and not args.epochs:
        paths = [run.DEFAULT_CHECKPOINT]
    if args.epochs:
        ckpt_dir = (args.checkpoint_dir or args.base_dir / "checkpoints" / args.experiment).resolve()
        found: dict[int, Path] = {}
        for q in ckpt_dir.glob("*.ckpt"):
            m = CKPT_RE.search(q.name)
            if m and (int(m.group(1)) not in found or q.stat().st_mtime > found[int(m.group(1))].stat().st_mtime):
                found[int(m.group(1))] = q
        missing = [e for e in args.epochs if e not in found]
        if missing:
            raise SystemExit(f"Epochs not found in {ckpt_dir}: {missing}")
        paths += [found[e] for e in args.epochs]
    ckpts = []
    for q in paths:
        q = q.resolve()
        if not q.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {q}")
        age = time.time() - q.stat().st_mtime
        if age < 120 and not args.allow_fresh and q != run.DEFAULT_CHECKPOINT.resolve():  # the shipped file is never mid-write; a clone just renews its mtime
            raise SystemExit(f"{q.name} was modified {age:.0f}s ago and may still be being written by a training run; wait or pass --allow-fresh.")
        data = q.read_bytes()  # hash and load the SAME bytes
        sha = hashlib.sha256(data).hexdigest()
        m = CKPT_RE.search(q.name)
        ckpts.append({"path": str(q), "label": f"{q.parent.name}/{q.stem}", "run": q.parent.name, "sha256": sha, "tag": f"{q.stem}__{sha[:10]}",
                      "epoch": int(m.group(1)) if m else None, "val_loss": float(m.group(2)) if m else None, "_bytes": data})
    return ckpts


def load_model_bytes(data: bytes, device: torch.device) -> Model:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    model = Model()
    inc = model.load_state_dict(state, strict=False)
    unexpected = [k for k in inc.unexpected_keys if not k.startswith("vgg.")]  # train.py's Model carries a frozen VGG; demo Model does not
    if inc.missing_keys or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing {inc.missing_keys[:5]}, unexpected {unexpected[:5]}")
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


# ----------------------------------------------------------------------------- data
def group_pairs(pairs: list[tuple[Path, Path, Path]]) -> tuple[dict, dict]:
    """Group pairs by content image (c-c{j}) so each content is decoded once; also collect the unique style images.
    The EvaluationSet stores one copy of the content per pair; equality across the group is checked by file size."""
    groups: dict[str, dict] = {}
    styles: dict[str, dict] = {}
    for c, s, rel in pairs:
        name = rel.with_suffix("").as_posix()
        m = PAIR_RE.fullmatch(name)
        ck, sk = (m.group(1), m.group(2)) if m else (name, name)
        g = groups.setdefault(ck, {"path": c, "size": c.stat().st_size, "items": []})
        st = styles.setdefault(sk, {"path": s, "size": s.stat().st_size})
        if c.stat().st_size != g["size"] or s.stat().st_size != st["size"]:
            raise RuntimeError(f"{name}: content/style file differs from others with the same key; grouping by filename is unsafe here")
        g["items"].append((sk, name))
    return groups, styles


def resize_max_pixels(img: torch.Tensor, max_pixels: int) -> torch.Tensor:
    """Truncated (int) target size, bilinear + antialias."""
    h, w = img.shape[-2:]
    if max_pixels <= 0 or h * w <= max_pixels:
        return img
    scale = (max_pixels / (h * w)) ** 0.5
    return F.interpolate(img, (int(h * scale), int(w * scale)), mode="bilinear", align_corners=False, antialias=True)


class ContentDataset(Dataset):
    """One item per content image: full-res uint8 (for the model, resized on the GPU) + the 1500x1500 LANCZOS cycle reference."""

    def __init__(self, groups: dict, order: list[str]):
        self.groups, self.order = groups, order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int) -> dict:
        ck = self.order[i]
        g = self.groups[ck]
        with Image.open(g["path"]) as im:
            im = im.convert("RGB")
            full = np.array(im)
            ref = np.array(im.resize((1500, 1500), Image.LANCZOS))
        return {"key": ck, "content": torch.from_numpy(full), "content1500": torch.from_numpy(ref), "items": g["todo"]}


def worker_init(_: int) -> None:
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------- metrics
def ycbcr_sorted(arr_u8: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 sRGB -> (3, H*W) per-channel sorted Y'CbCr values, float64 (model.py's matrix and clips)."""
    x = arr_u8.reshape(-1, 3).astype(np.float64) / 255.0 @ RGB2YCBCR.T
    x[:, 0] = x[:, 0].clip(0, 1)
    x[:, 1:] = x[:, 1:].clip(-0.5, 0.5)
    return np.sort(x.T, axis=1)


def color_cycle_metrics(out_u8: np.ndarray, cyc_u8: np.ndarray, style_sorted: np.ndarray, ref1500: np.ndarray) -> list[float]:
    o256 = np.array(Image.fromarray(out_u8).resize((256, 256), Image.LANCZOS))
    cm = ((ycbcr_sorted(o256) - style_sorted) ** 2).mean(1) * 255**2
    c1500 = np.array(Image.fromarray(cyc_u8).resize((1500, 1500), Image.LANCZOS))
    d = c1500.astype(np.int64) - ref1500.astype(np.int64)
    return [float(cm[0]), float(cm[1]), float(cm[2]), float((d * d).sum() / d.size)]  # x255^2 of /255 values == exact integer MSE


def jpeg_bytes(arr_u8: np.ndarray) -> bytes:  # PIL quality=95, default (4:2:0) subsampling, sRGB ICC
    b = io.BytesIO()
    Image.fromarray(arr_u8).save(b, format="JPEG", quality=95, icc_profile=SRGB_ICC)
    return b.getvalue()


class Inception:
    """pyiqa 'fid' feature extractor: clean-fid resize + pytorch-fid InceptionV3 (resize_input=False, normalize_input=False)."""

    def __init__(self, base_dir: Path, device: torch.device):
        from pytorch_fid.inception import InceptionV3

        cached = base_dir / "cache" / INCEPTION_NAME
        if not cached.is_file():
            cached.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading inception weights to {cached}")
            urllib.request.urlretrieve(INCEPTION_URL, cached)
        hub = Path(torch.hub.get_dir()) / "checkpoints" / INCEPTION_NAME  # where pytorch_fid looks; link our copy there
        if not hub.exists():
            hub.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(cached, hub)
        self.net = InceptionV3([3], resize_input=False, normalize_input=False, use_fid_inception=True).eval().requires_grad_(False).to(device)
        self.device = device

    @staticmethod
    def resize299(arr_u8: np.ndarray) -> np.ndarray:  # per-channel float32 bicubic, no re-quantisation
        chans = [np.asarray(Image.fromarray(arr_u8[:, :, c].astype(np.float32), mode="F").resize((299, 299), Image.BICUBIC)).clip(0, 255) for c in range(3)]
        return np.stack(chans, 0).astype(np.float32)

    @torch.inference_mode()
    def __call__(self, arr_u8: np.ndarray) -> np.ndarray:
        x = (torch.from_numpy(self.resize299(arr_u8))[None].to(self.device) - 128) / 128
        return self.net(x)[0].reshape(2048).float().cpu().numpy()


def frechet_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = f1.mean(0, dtype=np.float64), f2.mean(0, dtype=np.float64)
    s1, s2 = np.cov(f1.astype(np.float64), rowvar=False), np.cov(f2.astype(np.float64), rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID: imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def real_features(real_files: list[Path], real_dir: Path, inception: Inception, workers: int) -> np.ndarray:
    npy = real_dir / "inception_feats.npy"
    if npy.is_file():
        feats = np.load(npy)
        if feats.shape == (len(real_files), 2048):
            return feats
    real_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def one(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            return inception(np.array(im.convert("RGB")))

    with ThreadPoolExecutor(workers) as pool:
        feats = np.stack(list(pool.map(one, real_files)))
    np.save(npy, feats)
    (real_dir / "files.json").write_text(json.dumps([str(p) for p in real_files], indent=0))
    print(f"Real-set inception features: {len(real_files)} images in {time.perf_counter() - t0:.1f}s -> {npy}")
    return feats


def cluster_bootstrap(values: np.ndarray, clusters: list[str], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling CONTENT images (clusters) with replacement."""
    keys = sorted(set(clusters))
    idx = np.array([keys.index(c) for c in clusters])
    sums = np.bincount(idx, weights=values, minlength=len(keys))
    cnts = np.bincount(idx, minlength=len(keys)).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- evaluation of one checkpoint
def evaluate_checkpoint(ck: dict, out_dir: Path, groups: dict, styles: dict, pair_names: list[str], cfg_hash: str, device: torch.device,
                        inception: Inception, args: argparse.Namespace) -> None:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pid = lock_path.read_text().strip()
        if pid.isdigit() and Path(f"/proc/{pid}").exists():
            raise SystemExit(f"{out_dir} is being evaluated by pid {pid}; refusing to run twice on the same checkpoint.")
        print(f"  removing stale lock (pid {pid} is gone)")
        lock_path.write_text(str(os.getpid()))
    partial = out_dir / "metrics_partial.jsonl"
    feat_paths = {"jpeg": out_dir / "inception_feats.npy", "lossless": out_dir / "inception_feats_lossless.npy"}
    pcfg = out_dir / "partial_config.json"
    index = {n: i for i, n in enumerate(pair_names)}
    done: dict[str, dict] = {}
    if pcfg.is_file() and json.loads(pcfg.read_text()).get("config_hash") == cfg_hash and partial.is_file() and all(p.is_file() for p in feat_paths.values()):
        for line in partial.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["name"]] = r
            except ValueError:
                pass  # truncated last line of an interrupted run
        feats = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in feat_paths.items()}
    else:
        for p in (partial, pcfg, *feat_paths.values()):
            p.unlink(missing_ok=True)
        feats = {k: np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(len(pair_names), 2048)) for k, p in feat_paths.items()}
        pcfg.write_text(json.dumps({"config_hash": cfg_hash, "checkpoint": ck["path"], "sha256": ck["sha256"]}))
    for g in groups.values():
        g["todo"] = [(sk, name, index[name]) for sk, name in g["items"] if name not in done]
    order = [k for k, g in groups.items() if g["todo"]]
    n_todo = sum(len(groups[k]["todo"]) for k in order)
    print(f"  {len(done)} pairs cached, {n_todo} to compute over {len(order)} content images")
    ext = args.image_format
    if not args.no_images:
        (out_dir / "stylized").mkdir(exist_ok=True)
        (out_dir / "cycle").mkdir(exist_ok=True)

    model = load_model_bytes(ck["_bytes"], device)
    forward = lambda content, style_in: model(content, style_in)  # noqa: E731  Model.forward(source_srgb, target_srgb)
    lock = threading.Lock()
    log = partial.open("a")
    timings = {"loader_wait": 0.0, "main_loop": 0.0, "post_drain": 0.0}

    def post(job: dict) -> None:  # worker thread: cycle JPEG encode/decode, PIL resizes, numpy metrics, inception features, saving
        out_u8, out_jpg, out_dec, cyc_u8, name, sk, idx, ref1500 = job["out"], job["out_jpg"], job["out_dec"], job["cyc"], job["name"], job["sk"], job["idx"], job["ref1500"]
        cyc_jpg = jpeg_bytes(cyc_u8)
        cyc_dec = np.array(Image.open(io.BytesIO(cyc_jpg)).convert("RGB"))
        prim = color_cycle_metrics(out_dec, cyc_dec, styles[sk]["sorted"], ref1500)
        loss = color_cycle_metrics(out_u8, cyc_u8, styles[sk]["sorted"], ref1500)
        f_jpeg, f_loss = inception(out_dec), inception(out_u8)
        row = {"name": name, "content": name.split("_")[0], "style": sk, "height": int(out_u8.shape[0]), "width": int(out_u8.shape[1]),
               **dict(zip(("Y", "Cb", "Cr", "Cycle"), prim)), **dict(zip(("Y_lossless", "Cb_lossless", "Cr_lossless", "Cycle_lossless"), loss))}
        if not args.no_images:
            if ext == "jpg":
                (out_dir / "stylized" / f"{name}.jpg").write_bytes(out_jpg)
                (out_dir / "cycle" / f"{name}.jpg").write_bytes(cyc_jpg)
            else:
                Image.fromarray(out_u8).save(out_dir / "stylized" / f"{name}.png", compress_level=3, icc_profile=SRGB_ICC)
                Image.fromarray(cyc_u8).save(out_dir / "cycle" / f"{name}.png", compress_level=3, icc_profile=SRGB_ICC)
        with lock:
            feats["jpeg"][idx] = f_jpeg
            feats["lossless"][idx] = f_loss
            feats["jpeg"].flush()
            feats["lossless"].flush()
            log.write(json.dumps(row) + "\n")
            log.flush()
            done[name] = row

    to_u8 = lambda t: t[0].mul(255).round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()  # noqa: E731

    def cycle_pass(out_dec: np.ndarray, content256: torch.Tensor) -> torch.Tensor:
        """The DECODED stylized JPEG is the source, the 256x256 content the histogram target."""
        src = torch.from_numpy(out_dec).to(device).permute(2, 0, 1)[None].float().div_(255)
        return forward(src, content256)
    pending: deque = deque()
    loader = DataLoader(ContentDataset(groups, order), batch_size=None, shuffle=False, num_workers=min(args.workers, len(order)),
                        prefetch_factor=2, worker_init_fn=worker_init) if order else []
    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        n_now = 0
        t_wait = time.perf_counter()
        for item in loader:
            timings["loader_wait"] += time.perf_counter() - t_wait
            t0 = time.perf_counter()
            full = item["content"].to(device).permute(2, 0, 1)[None].float().div_(255)
            content256 = F.interpolate(full, (256, 256), mode="bilinear", align_corners=False, antialias=True)  # cycle-pass histogram target
            content = resize_max_pixels(full, args.max_pixels)
            del full
            ref1500 = item["content1500"].numpy()
            for sk, name, idx in item["items"]:
                out_u8 = to_u8(forward(content, styles[sk]["model_in"]))
                out_jpg = jpeg_bytes(out_u8)  # the saved file == what the cycle pass sees
                out_dec = np.array(Image.open(io.BytesIO(out_jpg)).convert("RGB"))
                cyc_u8 = to_u8(cycle_pass(out_dec, content256))
                pending.append(pool.submit(post, {"out": out_u8, "out_jpg": out_jpg, "out_dec": out_dec, "cyc": cyc_u8, "name": name, "sk": sk, "idx": idx, "ref1500": ref1500}))
                while len(pending) > 2 * args.workers:
                    pending.popleft().result()
            timings["main_loop"] += time.perf_counter() - t0
            n_now += len(item["items"])
            if n_now % 190 < len(item["items"]) or n_now == n_todo:
                print(f"  [{n_now}/{n_todo}] {item['key']}  elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
            t_wait = time.perf_counter()
        t0 = time.perf_counter()
        for f in pending:
            f.result()
        timings["post_drain"] = time.perf_counter() - t0
    log.close()

    # finalize: metrics.csv, FID, summary.json, done.json
    missing = [n for n in pair_names if n not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} pairs missing after evaluation (first: {missing[:3]})")
    rows = [done[n] for n in pair_names]
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    t0 = time.perf_counter()
    real = np.load(out_dir.parent / "_real" / "inception_feats.npy")
    fid, fid_loss = frechet_distance(np.asarray(feats["jpeg"]), real), frechet_distance(np.asarray(feats["lossless"]), real)
    timings["fid"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_start
    col = lambda k: np.array([r[k] for r in rows], dtype=np.float64)  # noqa: E731
    contents = [r["content"] for r in rows]
    summary = {k: v for k, v in ck.items() if k != "_bytes"}
    summary.update({"config_hash": cfg_hash, "metric_version": METRIC_VERSION, "n_pairs": len(rows), "n_contents": len(set(contents)), "n_real": int(real.shape[0]),
                    "FID": fid, "FID_lossless": fid_loss, "fid_subset": args.limit > 0, "tf32": args.tf32, "no_lowpass": args.no_lowpass,
                    "cycle_recipe": "model(decoded stylized JPEG, content 256x256 bilinear-antialias)", "ignore_icc": args.ignore_icc,
                    "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), "timings_s": {k: round(v, 1) for k, v in timings.items()},
                    "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__},
                    "images_saved": not args.no_images, "image_format": ext})
    for k in ("Y", "Cb", "Cr", "Cycle", "Y_lossless", "Cb_lossless", "Cr_lossless", "Cycle_lossless"):
        v = col(k)
        summary[k], summary[k + "_median"], summary[k + "_std"] = float(v.mean()), float(np.median(v)), float(v.std(ddof=1))
        summary[k + "_ci_lo"], summary[k + "_ci_hi"] = cluster_bootstrap(v, contents)
    summary["per_style"] = {sk: {k: float(np.mean([r[k] for r in rows if r["style"] == sk])) for k in ("Y", "Cb", "Cr", "Cycle")} for sk in sorted(styles)}
    atomic_write(out_dir / "summary.json", json.dumps(summary, indent=1))
    atomic_write(out_dir / "done.json", json.dumps({"config_hash": cfg_hash, "metric_version": METRIC_VERSION, "tag": ck["tag"], "sha256": ck["sha256"]}))
    for p in (partial, pcfg, lock_path):
        p.unlink(missing_ok=True)
    size = sum(p.stat().st_size for d in ("stylized", "cycle") for p in (out_dir / d).glob("*") if (out_dir / d).is_dir()) / 2**30
    print(f"  done in {timings['total']:.0f}s (main loop {timings['main_loop']:.0f}s incl. back-pressure, loader wait {timings['loader_wait']:.0f}s, "
          f"drain {timings['post_drain']:.0f}s, fid {timings['fid']:.1f}s); images on disk {size:.2f} GB")


def is_done(out_dir: Path, cfg_hash: str) -> bool:
    try:
        d = json.loads((out_dir / "done.json").read_text())
        return d.get("config_hash") == cfg_hash and d.get("metric_version") == METRIC_VERSION and (out_dir / "summary.json").is_file()
    except (OSError, ValueError):
        return False


# ----------------------------------------------------------------------------- table + figures
def sort_key(s: dict):
    return (s.get("run") or "", s["epoch"] if s.get("epoch") is not None else -1, s["label"])


def format_table(summaries: list[dict]) -> str:
    sub = any(s.get("fid_subset") for s in summaries)
    rows = [[s["label"], "" if s["epoch"] is None else str(s["epoch"]), "" if s["val_loss"] is None else f"{s['val_loss']:.2f}",
             *[f"{s[k]:.2f}" for k in COLS[:4]], f"{s['FID']:.2f}" + ("*" if s.get("fid_subset") else ""), str(s["n_pairs"]), f"{s['timings_s']['total']:.0f}"] for s in summaries]
    rows.append([REFERENCE_ROW["label"], "", "", *[f"{REFERENCE_ROW[k]:.2f}" for k in COLS], "", ""])
    head = ["checkpoint", "epoch", "val", "Y'", "Cb", "Cr", "Cycle", "FID", "n", "s"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    fmt = lambda r: "| " + " | ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))) + " |"  # noqa: E731
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [fmt(head), sep, *[fmt(r) for r in rows]]
    lines.append(f"\nlossless (uint8 in memory, no JPEG) minus primary (decoded JPEG q95 4:2:0), per checkpoint:")
    for s in summaries:
        lines.append(f"  {s['label']:<52} " + "  ".join(f"{k} {s[k + '_lossless'] - s[k]:+.2f}" for k in COLS) + f"   95% CI (content-clustered bootstrap): Y' [{s['Y_ci_lo']:.1f},{s['Y_ci_hi']:.1f}] Cycle [{s['Cycle_ci_lo']:.1f},{s['Cycle_ci_hi']:.1f}]")
    if sub:
        lines.append("* FID on a --limit subset: small sample, not comparable to full-set FIDs.")
    return "\n".join(lines)


def plot_summary(summaries: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(19, 5.2), constrained_layout=True)
    x = np.arange(len(summaries))
    labels = [f"{s['run']}\n{'ep ' + str(s['epoch']) if s['epoch'] is not None else s['label'].split('/')[-1]}" for s in summaries]
    for ax, k in zip(axes, COLS):
        y = np.array([s[k] for s in summaries])
        if k != "FID":
            ax.errorbar(x, y, yerr=[y - [s[k + "_ci_lo"] for s in summaries], [s[k + "_ci_hi"] for s in summaries] - y], color=C_BLUE, ls="none", marker="o", ms=6, capsize=3, label="this pipeline (95% CI, content-clustered)")
        else:
            ax.plot(x, y, color=C_BLUE, ls="none", marker="o", ms=6, label="this pipeline")
        ax.axhline(REFERENCE_ROW[k], color=C_ORANGE, ls="--", lw=1.2, label=f"README reference {REFERENCE_ROW[k]:.2f}")
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=C_TEXT2)
        ax.set_title(f"{k} (lower = better)", loc="left", color=C_TEXT, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, color=C_TEXT2)
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.grid(True, color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    axes[0].set_ylabel("MSE x 255^2", color=C_TEXT2, fontsize=8)
    fig.suptitle(f"Hist2Style metrics on EvaluationSet ({summaries[-1]['n_pairs']} pairs; dashed = README reference)", color=C_TEXT, fontsize=11)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_heatmap(s: dict, out_dir: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((out_dir / "metrics.csv").open()))
    contents = sorted({r["content"] for r in rows}, key=lambda c: int(c[3:]) if c[3:].isdigit() else 0)
    stylesk = sorted({r["style"] for r in rows}, key=lambda c: int(c[3:]) if c[3:].isdigit() else 0)
    fig, axes = plt.subplots(4, 1, figsize=(max(8, 0.12 * len(contents)), 2.2 * 4), constrained_layout=True)
    for ax, k in zip(axes, COLS[:4]):
        m = np.full((len(stylesk), len(contents)), np.nan)
        for r in rows:
            m[stylesk.index(r["style"]), contents.index(r["content"])] = float(r[k])
        im = ax.imshow(np.log10(np.maximum(m, 1e-3)), aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title(f"log10 {k} per pair  (mean {np.nanmean(m):.1f}, median {np.nanmedian(m):.1f}; rows = styles, cols = contents)", loc="left", fontsize=9, color=C_TEXT)
        ax.set_yticks(range(len(stylesk)))
        ax.set_yticklabels(stylesk, fontsize=5)
        ax.set_xticks(range(0, len(contents), max(1, len(contents) // 20)))
        ax.set_xticklabels([contents[i] for i in range(0, len(contents), max(1, len(contents) // 20))], fontsize=5, rotation=90)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle(s["label"], color=C_TEXT, fontsize=10)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_examples(s: dict, out_dir: Path, groups: dict, styles: dict, n: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not s.get("images_saved") or n <= 0:
        return
    rows = {r["name"]: r for r in csv.DictReader((out_dir / "metrics.csv").open())}
    picks = []
    for ck_ in sorted(groups)[:: max(1, len(groups) // n)]:
        sk, name = groups[ck_]["items"][len(picks) % len(groups[ck_]["items"])]
        picks.append((ck_, sk, name))
        if len(picks) == n:
            break
    ext = s["image_format"]

    def load(p: Path, side: int = 700) -> np.ndarray:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            return np.array(im.resize((side, max(1, int(side * h / w))) if w >= h else (max(1, int(side * w / h)), side), Image.BILINEAR))

    fig, axes = plt.subplots(len(picks), 5, figsize=(16, 3.4 * len(picks)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, (ck_, sk, name) in enumerate(picks):
        content = load(groups[ck_]["path"])
        cyc = load(out_dir / "cycle" / f"{name}.{ext}")
        cyc = np.array(Image.fromarray(cyc).resize((content.shape[1], content.shape[0]), Image.BILINEAR))
        diff = np.abs(cyc.astype(np.float32) - content.astype(np.float32)).mean(-1)
        r = rows[name]
        panels = [("content", content), ("style (histogram source)", load(styles[sk]["path"])),
                  (f"stylized  Y' {float(r['Y']):.0f} Cb {float(r['Cb']):.0f} Cr {float(r['Cr']):.0f}", load(out_dir / "stylized" / f"{name}.{ext}")), (f"cycle  MSE {float(r['Cycle']):.0f}", cyc)]
        for j, (title, im) in enumerate(panels):
            axes[i, j].imshow(im)
            axes[i, j].set_title(title, fontsize=8, color=C_TEXT)
        h = axes[i, 4].imshow(diff, cmap="Blues", vmin=0, vmax=40)
        axes[i, 4].set_title(f"|cycle - content| (mean {diff.mean():.1f}/255)", fontsize=8, color=C_TEXT2)
        axes[i, 0].set_ylabel(name, fontsize=8, color=C_TEXT2)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(h, ax=axes[:, 4], fraction=0.04, label="|diff| / 255 (clipped at 40)")
    fig.suptitle(f"{s['label']}  (Y' {s['Y']:.1f}  Cb {s['Cb']:.1f}  Cr {s['Cr']:.1f}  Cycle {s['Cycle']:.1f}  FID {s['FID']:.1f})", color=C_TEXT, fontsize=10)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    t_main = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = False
    if args.no_lowpass:
        model_module.lowpass = lambda x, *a, **k: x
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpts = resolve_checkpoints(args)

    evaluation_dir = args.evaluation_dir.resolve() if args.evaluation_dir else run.download_and_extract(args.evaluation_url, (args.data_dir / "EvaluationSet").resolve())
    content_dir, style_dir = run.resolve_splits(evaluation_dir, args.content_dir, args.style_dir)
    all_pairs = run.pair_evaluation_images(content_dir, style_dir)
    pairs = all_pairs if args.limit == 0 else run.random.Random(args.seed).sample(all_pairs, min(args.limit, len(all_pairs)))
    pairs = sorted(pairs, key=lambda p: p[2].as_posix())
    pair_names = [p[2].with_suffix("").as_posix() for p in pairs]
    groups, styles = group_pairs(pairs)
    all_groups, all_styles = group_pairs(all_pairs)
    real_files = [all_groups[k]["path"] for k in sorted(all_groups)] + [all_styles[k]["path"] for k in sorted(all_styles)]
    ds_id = hashlib.sha256(json.dumps([(p.name, p.stat().st_size) for p in real_files]).encode()).hexdigest()[:6]
    cfg_name = ("all" if args.limit == 0 else f"n{args.limit}-seed{args.seed}") + f"-mp{args.max_pixels}-ds{ds_id}" + ("-tf32" if args.tf32 else "") + ("-nolp" if args.no_lowpass else "")
    cfg_hash = hashlib.sha256(json.dumps({"pairs": pair_names, "max_pixels": args.max_pixels, "ds": ds_id, "tf32": args.tf32, "no_lowpass": args.no_lowpass, "metric_version": METRIC_VERSION}).encode()).hexdigest()[:16]
    table_dir = args.base_dir / "eval_table" / cfg_name
    table_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {[c['label'] for c in ckpts]}\nPairs: {len(pairs)} ({len(groups)} contents x {len(styles)} styles; limit={args.limit}, seed={args.seed}) from {evaluation_dir}\n"
          f"Real set for FID: {len(real_files)} unique images (dataset id {ds_id})\nDevice: {device} = {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}); TF32 {'ON' if args.tf32 else 'off'}; grid low-pass {'OFF' if args.no_lowpass else 'on'}\n"
          f"Cycle: model(decoded stylized JPEG, content 256x256 bilinear-aa) by the checkpoint under test\nOutputs: {table_dir}\nConfig hash: {cfg_hash}")

    todo = [c for c in ckpts if args.overwrite or not is_done(table_dir / c["tag"], cfg_hash)]
    if todo:
        if args.ignore_icc:
            print("NOTE: --ignore-icc: embedded colour profiles are ignored; raw pixel values are used.")
        else:
            run.validate_srgb_images([p for c, s, _ in pairs for p in (c, s)])
        inception = Inception(args.base_dir, device)
        real_features(real_files, table_dir / "_real", inception, args.workers)
        # style images are shared by all pairs: model input (256x256 bilinear+antialias) and metric reference (256x256 LANCZOS, sorted Y'CbCr)
        with torch.inference_mode():
            for sk, st in styles.items():
                with Image.open(st["path"]) as im:
                    im = im.convert("RGB")
                    full = torch.from_numpy(np.array(im)).to(device).permute(2, 0, 1)[None].float().div_(255)
                    st["model_in"] = F.interpolate(full, (256, 256), mode="bilinear", align_corners=False, antialias=True)
                    st["sorted"] = ycbcr_sorted(np.array(im.resize((256, 256), Image.LANCZOS)))
                    del full
        for c in todo:
            out_dir = table_dir / c["tag"]
            if args.overwrite and out_dir.exists():
                shutil.rmtree(out_dir)
            print(f"\n=== {c['label']}  sha256 {c['sha256'][:10]}  -> {out_dir}")
            evaluate_checkpoint(c, out_dir, groups, styles, pair_names, cfg_hash, device, inception, args)

    listed = {c["tag"] for c in ckpts}
    if args.table_all:
        listed |= {p.parent.name for p in table_dir.glob("*/done.json") if is_done(p.parent, cfg_hash)}
    summaries = sorted((json.loads((table_dir / t / "summary.json").read_text()) for t in listed if is_done(table_dir / t, cfg_hash)), key=sort_key)
    if not summaries:
        raise SystemExit("Nothing evaluated for this config.")
    table = format_table(summaries)
    atomic_write(table_dir / "table.md", table + "\n")
    with (table_dir / "table.csv").open("w", newline="") as f:
        fields = ["label", "run", "epoch", "val_loss", *COLS, *[k + "_lossless" for k in COLS], *[k + s for k in COLS[:4] for s in ("_ci_lo", "_ci_hi", "_median")],
                  "n_pairs", "n_real", "fid_subset", "tf32", "no_lowpass", "ignore_icc", "tag", "sha256", "path", "evaluated_at"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    plot_summary(summaries, table_dir / "summary.png")
    for s in summaries:
        if s["tag"] in {c["tag"] for c in ckpts}:
            plot_heatmap(s, table_dir / s["tag"], table_dir / f"heatmap_{s['tag']}.png")
            plot_examples(s, table_dir / s["tag"], groups, styles, args.examples, table_dir / f"examples_{s['tag']}.png")
    print(f"\nHist2Style metrics (lower = better; Y'/Cb/Cr/Cycle = MSE x 255^2 over {summaries[-1]['n_pairs']} pairs, measured from the saved JPEG q95 4:2:0 outputs; "
          f"FID vs {summaries[-1]['n_real']} real images, compare within one table)\n{table}")
    print(f"\nTable + figures: {table_dir}\nTotal runtime: {time.perf_counter() - t_main:.1f}s")


if __name__ == "__main__":
    main()
