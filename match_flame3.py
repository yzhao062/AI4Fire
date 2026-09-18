"""Map the 408 WildFireVQA items to their FLAME 3 images on disk, by thermal fingerprint, and render the thermal view.

The item set carries WildFireVQA's per-unit image paths (Sycan 2A 00001-00364 fire, Sycan 2D 00001-00258 fire and
00001-00116 no fire), while the FLAME 3 computer-vision subset on disk is numbered flat (Fire 00001-00622, No Fire
00001-00116). Rather than guess the concatenation order, each item is matched by the fingerprint its own temp_summary
carries: min, max, and mean of the Celsius TIFF to three decimals, with the standard deviation as a check (WildFireVQA
used the population form, ddof 0, which this script confirms rather than assumes).

The thermal image the models see is rendered here from the Celsius TIFF with matplotlib's inferno colormap, scaled
from the frame's own minimum to its maximum, which is what WildFireVQA's prompt describes ("rendered using an inferno
colormap of radiometric TIFF", with the provided min and max as the endpoints of the scale). The camera's own JPG is
kept beside it in the map for reference and is not sent.

    python match_flame3.py            # write task-wildfirevqa/image-map.json and data/flame3/rendered/<uid>.jpg
"""
import json
import pathlib

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps

S = pathlib.Path(__file__).parent
TASK = S / "task-wildfirevqa"
ROOT = S / "data" / "flame3" / "FLAME 3 CV Dataset (Sycan Marsh)"
RENDER = S / "data" / "flame3" / "rendered"


def stats(path):
    a = np.array(Image.open(path), dtype=np.float64)
    return (round(float(a.min()), 3), round(float(a.max()), 3), round(float(a.mean()), 3),
            round(float(a.std(ddof=0)), 3), round(float(a.std(ddof=1)), 3))


def render(tiff, out, vmin, vmax):
    a = np.array(Image.open(tiff), dtype=np.float64)
    x = np.clip((a - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgb = (colormaps["inferno"](x)[..., :3] * 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out, quality=92)


def main():
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    index = {}
    for state in ("Fire", "No Fire"):
        for tiff in sorted((ROOT / state / "Thermal" / "Celsius TIFF").glob("*.TIFF")):
            mn, mx, mean, s0, s1 = stats(tiff)
            index.setdefault((mn, mx, mean), []).append((state, tiff.stem, s0, s1))
    print("tiffs indexed:", sum(len(v) for v in index.values()), "| distinct (min, max, mean):", len(index))

    image_map, ddof_seen, unmatched, ambiguous = {}, {0: 0, 1: 0}, [], []
    for it in items:
        uid = it["image"]["image_uid"]
        if uid in image_map:
            continue
        ts = it["temp_summary"]
        key = (round(ts["min"], 3), round(ts["max"], 3), round(ts["mean"], 3))
        hits = index.get(key, [])
        hits = [h for h in hits if round(ts["std"], 3) in (h[2], h[3])]
        if not hits:
            # One image (Sycan 2D fire 00168) has a mean that rounds to 18.308 in the item and 18.309 here, a
            # last-digit difference from summation order; accept a unique frame that agrees on min and max
            # exactly and on mean and std within one unit of the third decimal.
            hits = [h for k, hs in index.items() if k[0] == key[0] and k[1] == key[1] and abs(k[2] - key[2]) <= 0.0015
                    for h in hs if abs(round(ts["std"], 3) - h[2]) <= 0.0015]
        if not hits:
            unmatched.append(uid)
            continue
        if len(hits) > 1:
            ambiguous.append((uid, hits))
            continue
        state, stem, s0, s1 = hits[0]
        ddof_seen[0 if round(ts["std"], 3) == s0 else 1] += 1
        rel = lambda *p: str(pathlib.Path(state, *p, stem + ".JPG" if p[-1] != "Celsius TIFF" else stem + ".TIFF").as_posix())
        rendered = RENDER / (uid.replace("/", "_").replace(" ", "") + ".jpg")
        render(ROOT / state / "Thermal" / "Celsius TIFF" / (stem + ".TIFF"), rendered, ts["min"], ts["max"])
        image_map[uid] = {
            "burn_state_on_disk": state, "disk_id": stem,
            "rgb_corrected_fov": rel("RGB", "Corrected FOV"),
            "thermal_camera_jpg": rel("Thermal", "Raw JPG"),
            "thermal_celsius_tiff": rel("Thermal", "Celsius TIFF"),
            "thermal_rendered_inferno": rendered.relative_to(S).as_posix(),
            "fingerprint": {"min": key[0], "max": key[1], "mean": key[2], "std": round(ts["std"], 3)},
        }
    print("items: %d | distinct images: %d | matched: %d | unmatched: %d | ambiguous: %d | std ddof seen: %s"
          % (len(items), len({i["image"]["image_uid"] for i in items}), len(image_map), len(unmatched), len(ambiguous), ddof_seen))
    for u in unmatched[:10]:
        print("  unmatched:", u)
    for u, h in ambiguous[:10]:
        print("  ambiguous:", u, h)
    (TASK / "image-map.json").write_text(json.dumps(image_map, indent=1), encoding="utf-8")
    print("wrote", TASK / "image-map.json")


if __name__ == "__main__":
    main()
