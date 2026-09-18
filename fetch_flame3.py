"""Fetch the FLAME 3 computer-vision subset (Sycan Marsh) that the WildFireVQA items are built on.

Two sources hold the same images. The canonical record is FLAME 3 on IEEE DataPort (DOI 10.21227/w0mz-aq48,
CVSubset.zip, 6.23 GB), whose download sits behind an IEEE login that goes through IEEE's single sign-on. The
Kaggle mirror (brycehopkins/flame-3-computer-vision-subset-sycan-marsh, 7.54 GB) is the same burn behind a
Kaggle API token, which is scriptable, so this script takes the mirror and records the DataPort DOI as the
source of the images. Credentials come from the gitignored .env in the repository root (KAGGLE_USERNAME and
KAGGLE_KEY); nothing here prints them.

    python fetch_flame3.py            # download the zip into data/flame3/ (resumable by rerunning)
    python fetch_flame3.py --unzip    # download if missing, then extract beside the zip
"""
import argparse
import os
import pathlib
import subprocess
import sys
import zipfile

S = pathlib.Path(__file__).parent
DEST = S / "data" / "flame3"
DATASET = "brycehopkins/flame-3-computer-vision-subset-sycan-marsh"
ZIP = DEST / "flame-3-computer-vision-subset-sycan-marsh.zip"


def load_env(path=S / ".env"):
    """Put the KEY=VALUE lines of .env into the process environment without echoing any value."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unzip", action="store_true", help="extract the zip after downloading")
    args = ap.parse_args()

    load_env()
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if not os.environ.get(k):
            sys.exit(f"{k} is not set; put it in .env")
    DEST.mkdir(parents=True, exist_ok=True)

    if not ZIP.exists():
        print("downloading", DATASET, "to", DEST, flush=True)
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET, "-p", str(DEST)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"kaggle download failed with exit code {r.returncode}")
    zips = sorted(DEST.glob("*.zip"))
    if not zips:
        sys.exit("no zip landed in " + str(DEST))
    z = zips[0]
    print("zip:", z.name, "%.2f GB" % (z.stat().st_size / 1e9), flush=True)

    if args.unzip:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            print("entries:", len(names), flush=True)
            zf.extractall(DEST)
        print("extracted to", DEST, flush=True)


if __name__ == "__main__":
    main()
