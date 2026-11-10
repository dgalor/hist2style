"""Download the Hist2Style data into ./data, where eval.py and train.py read it (they also download what they need on first use).

    python3 download_data.py            # everything (~65 GB)
    python3 download_data.py --eval     # evaluation set only
    python3 download_data.py --train    # training data only

  data/EvaluationSet/          content and style photos of the evaluation pairs    11.6 GB   Unsplash License
  data/unsplash25k_512_srgb/   unstylized training photos (512 px)                   0.7 GB   Unsplash License
  data/flux_style_ds_srgb/     stylized training images (FLUX.1 Kontext [dev])      52.8 GB   FLUX.1 [dev] Non-Commercial License

Each archive is extracted while it downloads into data/.<name>.partial/ and moved into place only when complete, so an interrupted
download never counts as finished; rerun the script to retry."""
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

BOX = "https://berkeley.box.com/shared/static/"
SETS = {  # folder name: (archive URL, size, part)
    "EvaluationSet": (BOX + "992k1bvsmr8wioicpnmmonvrclupkwvq.tar", "11.6 GB", "eval"),
    "unsplash25k_512_srgb": (BOX + "7wnwm0lme880vmsmu4xespn5uoz0bbzl", "0.7 GB", "train"),
    "flux_style_ds_srgb": (BOX + "utl9vmomwk0vg2l0ngusyl0ri5ob1way", "52.8 GB", "train"),
}


class Progress:
    """File wrapper that prints how much has been downloaded, every GB."""

    def __init__(self, f):
        self.f, self.n, self.next = f, 0, 1e9

    def read(self, size=-1):
        b = self.f.read(size)
        self.n += len(b)
        if self.n >= self.next:
            print(f"  {self.n / 1e9:.0f} GB", flush=True)
            self.next += 1e9
        return b


def fetch(name: str, data_dir: Path) -> None:
    url, size, _ = SETS[name]
    target = data_dir / name
    if target.is_dir():
        print(f"{name}: already in {target}")
        return
    partial = data_dir / f".{name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    print(f"{name}: downloading and extracting {size} from {url}", flush=True)
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=Progress(response), mode="r|*") as tar:
        tar.extractall(partial, filter="data")
    if not (partial / name).is_dir():
        raise RuntimeError(f"the archive did not contain {name}/ (left in {partial})")
    (partial / name).rename(target)
    shutil.rmtree(partial)
    print(f"{name}: done -> {target}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="store_true", help="evaluation set only")
    p.add_argument("--train", action="store_true", help="training data only")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data", help="default: ./data next to this file")
    args = p.parse_args()
    parts = {part for part, on in (("eval", args.eval), ("train", args.train)) if on} or {"eval", "train"}
    for name, (_, _, part) in SETS.items():
        if part in parts:
            fetch(name, args.data_dir)


if __name__ == "__main__":
    main()
