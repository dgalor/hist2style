import csv
import gzip
import os
from pathlib import Path
from typing import Mapping


ScoreCache = dict[tuple[str, str], float]


def load_score_cache(path: str | Path) -> ScoreCache:
    """Load a gzip-compressed, version-independent CSV score cache."""
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as cache_file:
        reader = csv.reader(cache_file)
        header = next(reader, None)
        if header != ["image_name", "style_name", "score"]:
            raise ValueError(f"Unexpected score-cache header in {path}: {header!r}")
        return {
            (image_name, style_name): float(score)
            for image_name, style_name, score in reader
        }


def save_score_cache(
    path: str | Path,
    scores: Mapping[tuple[str, str], float],
) -> None:
    """Atomically save scores as gzip-compressed CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        with gzip.open(
            temporary_path,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=9,
        ) as cache_file:
            writer = csv.writer(cache_file)
            writer.writerow(["image_name", "style_name", "score"])
            writer.writerows(
                (image_name, style_name, score)
                for (image_name, style_name), score in scores.items()
            )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
