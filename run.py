#!/usr/bin/env python3
"""Run a Hist2Style checkpoint on content/style image pairs.

    python3 run.py --content photo.jpg --style painting.jpg     # one pair          -> outputs/photo__painting.png
    python3 run.py --content contents/ --style styles/          # two folders, paired by identical filenames -> outputs/<name>.png

Inputs must be sRGB: an embedded non-sRGB ICC profile is an error; images without a profile are treated as sRGB. Outputs are lossless
PNG at the (downsampled, see --max-pixels) content resolution. The default checkpoint is the released demo/model-epoch=1500-val_loss=2.79.ckpt.
"""

from __future__ import annotations

import argparse
import io
import math
import random
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageCms, ImageOps
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "demo"))
from model import Model, batch_histogram_1D, srgb_to_ycbcr  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "demo" / "model-epoch=1500-val_loss=2.79.ckpt"
DEFAULT_DATA_DIR = ROOT / "data"
LOCAL_EVAL_ARCHIVE = ROOT / "EvaluationSet.tar"
DEFAULT_EVAL_URL = (
    "https://berkeley.box.com/shared/static/"
    "992k1bvsmr8wioicpnmmonvrclupkwvq.tar"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", type=Path, required=True, help="A content image, or a folder of content images.")
    parser.add_argument("--style", type=Path, required=True, help="A style image, or a folder of style images with the same filenames as --content.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Where the PNG outputs go (default: %(default)s).")
    parser.add_argument("--device", default="auto", help="Torch device, such as cuda:0, mps, or cpu (default: auto).")
    parser.add_argument("--max-pixels", type=int, default=2_000_000, help="Downsample content images larger than this; 0 disables downsampling.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_pairs(content: Path, style: Path) -> list[tuple[Path, Path, Path]]:
    """(content, style, output name) triples: one pair of files, or two folders paired by identical relative filenames."""
    if content.is_file() and style.is_file():
        return [(content, style, Path(f"{content.stem}__{style.stem}"))]
    if content.is_dir() and style.is_dir():
        pairs = pair_evaluation_images(content, style)
        if not pairs:
            raise FileNotFoundError(f"no images in {content} / {style}")
        return pairs
    raise ValueError("--content and --style must both be image files, or both be folders of images with matching filenames")


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            and not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def validate_srgb_images(paths: list[Path]) -> None:
    """Refuse images with an embedded non-sRGB profile.

    Images without an embedded ICC profile are treated as sRGB, following the
    convention used by common image viewers and web image pipelines.
    """
    errors: list[str] = []
    rgb_compatible_modes = {"1", "L", "LA", "P", "RGB", "RGBA"}

    print(f"Checking color profiles for {len(paths)} images")
    for path in paths:
        try:
            with Image.open(path) as image:
                if image.mode not in rgb_compatible_modes:
                    errors.append(f"{path}: image mode is {image.mode}, not RGB")
                    continue

                icc_bytes = image.info.get("icc_profile")
                if not icc_bytes:
                    continue

                try:
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                    identity = " ".join(
                        (
                            ImageCms.getProfileName(profile),
                            ImageCms.getProfileDescription(profile),
                        )
                    ).lower()
                except Exception as error:
                    errors.append(f"{path}: unreadable ICC profile ({error})")
                    continue

                if "srgb" not in identity:
                    profile_name = " ".join(identity.split()) or "unknown"
                    errors.append(
                        f"{path}: embedded profile is not sRGB ({profile_name})"
                    )
        except (OSError, ValueError) as error:
            errors.append(f"{path}: could not inspect image ({error})")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(
            f"Refusing to run: {len(errors)} image(s) are not verified as sRGB; convert them to sRGB first "
            f"(images without an embedded profile are treated as sRGB):\n{details}"
        )
    print("Color profile check passed (missing profiles treated as sRGB)")


def download_and_extract(url: str, destination: Path) -> Path:
    if destination.is_dir() and image_files(destination):
        print(f"Using existing evaluation set: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = not LOCAL_EVAL_ARCHIVE.is_file()
    archive = (
        destination.parent / "EvaluationSet.tar"
        if downloaded
        else LOCAL_EVAL_ARCHIVE
    )
    if downloaded:
        partial = archive.with_suffix(".tar.part")
        print(f"Downloading EvaluationSet from {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            partial.replace(archive)
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download EvaluationSet. Pass a working URL with "
                "--evaluation-url, or point --evaluation-dir at an extracted copy."
            ) from error
    else:
        print(f"Using local sRGB evaluation archive: {archive}")

    print(f"Extracting {archive}")
    try:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(destination.parent, filter="data")
    except (OSError, tarfile.TarError) as error:
        raise RuntimeError(f"Failed to extract {archive}: {error}") from error
    finally:
        if downloaded:
            archive.unlink(missing_ok=True)

    if not destination.is_dir():
        candidates = [
            path
            for path in destination.parent.iterdir()
            if path.is_dir() and image_files(path)
        ]
        if len(candidates) == 1:
            candidates[0].rename(destination)

    if not destination.is_dir() or not image_files(destination):
        raise RuntimeError(
            f"The archive did not create a usable evaluation set at {destination}"
        )
    return destination


def find_split(root: Path, kind: str) -> Path | None:
    aliases = {
        "content": ("content", "contents", "content_images", "contentimages"),
        "style": ("style", "styles", "style_images", "styleimages"),
    }[kind]
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]

    for directory in directories:
        normalized = directory.name.lower().replace("-", "_").replace(" ", "_")
        if normalized in aliases and image_files(directory):
            return directory
    for directory in directories:
        if kind in directory.name.lower() and image_files(directory):
            return directory
    return None


def resolve_splits(
    evaluation_dir: Path,
    content_override: Path | None,
    style_override: Path | None,
) -> tuple[Path, Path]:
    content_dir = content_override or find_split(evaluation_dir, "content")
    style_dir = style_override or find_split(evaluation_dir, "style")

    if content_dir is None or style_dir is None:
        populated = [
            path
            for path in evaluation_dir.iterdir()
            if path.is_dir() and image_files(path)
        ]
        if len(populated) == 2:
            populated.sort(key=lambda path: len(image_files(path)))
            style_dir = style_dir or populated[0]
            content_dir = content_dir or populated[1]

    if content_dir is None or style_dir is None:
        raise RuntimeError(
            "Could not identify the content and style folders. "
            "Specify them with --content-dir and --style-dir."
        )
    return content_dir.resolve(), style_dir.resolve()


def pair_evaluation_images(
    content_dir: Path, style_dir: Path
) -> list[tuple[Path, Path, Path]]:
    """Pair content and style images by their relative filename."""
    content_by_name = {
        path.relative_to(content_dir): path for path in image_files(content_dir)
    }
    style_by_name = {
        path.relative_to(style_dir): path for path in image_files(style_dir)
    }
    content_names = set(content_by_name)
    style_names = set(style_by_name)
    if content_names != style_names:
        missing_styles = sorted(content_names - style_names, key=lambda path: path.as_posix())
        missing_contents = sorted(style_names - content_names, key=lambda path: path.as_posix())
        details = []
        if missing_styles:
            details.append(
                "missing styles for: "
                + ", ".join(path.as_posix() for path in missing_styles[:5])
            )
        if missing_contents:
            details.append(
                "missing content for: "
                + ", ".join(path.as_posix() for path in missing_contents[:5])
            )
        raise RuntimeError(
            "content and style folders are not paired by filename (" + "; ".join(details) + ")"
        )

    return [
        (content_by_name[name], style_by_name[name], name)
        for name in sorted(content_names, key=lambda path: path.as_posix())
    ]


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_performance() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True


def load_model(checkpoint: Path, device: torch.device) -> Model:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = Model()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_model_keys = [
        key for key in incompatible.missing_keys if key.startswith("model.")
    ]
    if missing_model_keys:
        preview = ", ".join(missing_model_keys[:5])
        raise RuntimeError(
            f"Checkpoint is missing {len(missing_model_keys)} model weights "
            f"(first keys: {preview})"
        )
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=torch.float32)


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return pil_to_tensor(image).unsqueeze(0).to(device).float().div_(255.0)


def downsample(image: torch.Tensor, max_pixels: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    if max_pixels <= 0 or height * width <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (height * width))
    shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    return F.interpolate(
        image, size=shape, mode="bilinear", align_corners=False, antialias=True
    )


@torch.inference_mode()
def style_histogram(style_path: Path, device: torch.device) -> torch.Tensor:
    style = load_rgb(style_path, device)
    return batch_histogram_1D(srgb_to_ycbcr(style))


@torch.inference_mode()
def stylize(
    model: Model,
    content_path: Path,
    target_histogram: torch.Tensor,
    device: torch.device,
    max_pixels: int,
) -> Image.Image:
    content = downsample(load_rgb(content_path, device), max_pixels)
    content_histogram = batch_histogram_1D(srgb_to_ycbcr(content))
    output, _ = model.model(content, content_histogram, target_histogram)
    return to_pil_image(output[0].clamp(0, 1).cpu())


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.content, args.style)
    validate_srgb_images(sorted({path for content_path, style_path, _ in pairs for path in (content_path, style_path)}))

    configure_performance()
    device = select_device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(pairs)
    print(f"Checkpoint: {args.checkpoint.resolve()}\nPairs: {total}\nDevice: {device}\nWriting results to {args.output_dir.resolve()}")

    histograms: dict[Path, torch.Tensor] = {}
    for completed, (content_path, style_path, relative_path) in enumerate(pairs, 1):
        if style_path not in histograms:
            histograms[style_path] = style_histogram(style_path, device)
        output_path = (args.output_dir / relative_path).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output_path.exists():
            stylize(model, content_path, histograms[style_path], device, args.max_pixels).save(output_path)
        print(f"[{completed}/{total}] {output_path}")
    print(f"Done: {total} images")


if __name__ == "__main__":
    main()
