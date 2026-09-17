"""Download the Mesogeos Track A machine learning track into data/mesogeos.

Mesogeos (kondylatos2023mesogeos, NeurIPS 2023 Datasets and Benchmarks) hosts its data in a public Google
Drive folder, id 1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9. That folder holds mesogeos_cube.zarr, which is the
648 GB datacube, next to ml_tracks, which holds the two extracted machine-learning tracks. This script
takes only ml_tracks/a.danger_forecasting, four files totalling about 0.9 GB, and never touches the cube.

File ids resolved from the folder listing on 2026-09-15 through the embeddedfolderview endpoint, which is
the same listing route the public web view uses. They are pinned here so a rerun fetches the same bytes.
License: CC BY 4.0.
"""
import hashlib
import pathlib
import sys

import httpx

FOLDER = "1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9"
TRACK_A = "1dRyn7EAwG88f0QMKGz74rWY8krKH8VSV"
FILES = {
    "positives.csv": "1IYONBanlerMi84wedto-Vck8vmfnyVUR",
    "negatives.csv": "1qB6TjMCgpVvM04ysCZSgE-5sjJ2C9gNJ",
    "norms.json": "1uXsRpCJa7JNlPXmHvzCCIxs3E0eC6NUE",
    "vars_dict.json": "1PdwYyMX-51-zJ1dr36g8Hq-_7K7KTVAx",
}
OUT = pathlib.Path(__file__).parent / "data" / "mesogeos"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(fid, dest):
    # Drive answers a file over about 100 MB with an interstitial HTML page rather than the bytes. The
    # download is resumed by reposting the form fields that page carries, which is why this goes through a
    # session and inspects the first chunk rather than streaming straight to disk.
    with httpx.Client(headers=UA, follow_redirects=True, timeout=120) as c:
        r = c.get("https://drive.usercontent.google.com/download",
                  params={"id": fid, "export": "download", "confirm": "t"})
        if "text/html" in r.headers.get("content-type", ""):
            import re
            form = re.search(r'action="([^"]+)"', r.text)
            fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', r.text))
            if not form:
                raise RuntimeError("Drive returned HTML with no download form for %s" % fid)
            r = c.get(form.group(1).replace("&amp;", "&"), params=fields)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest.stat().st_size


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, fid in FILES.items():
        dest = OUT / name
        if dest.exists():
            size = dest.stat().st_size
            print("have   %-16s %12d bytes" % (name, size))
        else:
            size = fetch(fid, dest)
            print("pulled %-16s %12d bytes" % (name, size))
        total += size
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
        print("       sha256[:16] %s" % digest)
    print("total %.2f GB in %s" % (total / 1e9, OUT))
    sys.exit(0)
