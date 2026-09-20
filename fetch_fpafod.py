"""Download FPA-FOD 6th Edition (Short 2022, RDS-2013-0009.6) SQLite build and record the fetch.

Source page: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
Citation required by the archive:
  Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020
  [FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive.
  https://doi.org/10.2737/RDS-2013-0009.6
Use terms: "collected using funding from the U.S. Government and can be used without additional
permissions or fees", citation required.
"""
import hashlib
import json
import pathlib
import time
import zipfile

import httpx

URL = "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.6/RDS-2013-0009.6_Data_Format4_SQLITE.zip"
D = pathlib.Path(__file__).parent / "data" / "tooluse"
D.mkdir(parents=True, exist_ok=True)
ZIP = D / "RDS-2013-0009.6_Data_Format4_SQLITE.zip"
UA = {"User-Agent": "fire-bench/0.1 (AI4Fire benchmark; research evaluation)"}

if not ZIP.exists():
    t0 = time.time()
    with httpx.stream("GET", URL, headers=UA, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with ZIP.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    print("downloaded %.1f MB in %.0f s" % (ZIP.stat().st_size / 1e6, time.time() - t0))
else:
    print("zip already present: %.1f MB" % (ZIP.stat().st_size / 1e6))

sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    print("members:", len(names))
    for n in names:
        print("  %-60s %12d" % (n, z.getinfo(n).file_size))
    for n in names:
        if n.lower().endswith((".sqlite", ".db", ".html", ".xml", ".txt")):
            out = D / pathlib.Path(n).name
            if not out.exists():
                with z.open(n) as src, out.open("wb") as dst:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        dst.write(b)
                print("extracted", out.name, out.stat().st_size)

(D / "fetch.json").write_text(json.dumps({
    "url": URL,
    "sha256": sha,
    "bytes": ZIP.stat().st_size,
    "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "citation": "Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 "
                "[FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive. "
                "https://doi.org/10.2737/RDS-2013-0009.6",
}, indent=2), encoding="utf-8")
print("sha256", sha)
