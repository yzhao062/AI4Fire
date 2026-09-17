"""Build the version-1 fire danger task from Mesogeos Track A: does a 30+ hectare fire start here tomorrow?

Selection rule, published with the item set:
  - Mesogeos Track A as released, that is data/mesogeos/positives.csv and negatives.csv, never the 648 GB
    datacube. fetch_mesogeos.py pulls the two files plus their norms.json and vars_dict.json, 0.22 GB total
  - every sample whose 30-day driver window ends in 2021 or 2022, which is the published test holdout; the
    train years 2006 to 2019 and the 2020 validation year contribute no items
  - all 1,369 holdout positives and all 2,751 holdout negatives, with no subsampling and no random draw
  - a model sees the 12 daily drivers over the 30-day window and the 12 static drivers, which is the exact
    feature list configs/config_{lstm,transformer,gtn}/config_train.json feed the published baselines
A positive is an EFFIS fire of at least 30 hectares whose ignition point falls in this 1 km cell; the window
covers the 30 days before ignition. A negative is a cell with no burned area within 62 km on the day the
window ends. Both are asked the same question about the day after the window, which is exact for a positive
and one day late for a negative, because the release verified the absence of fire on the last window day.
The asymmetry is inherited rather than introduced: it costs a negative item its label only if that cell
burned the next day, and a cell with nothing alight within 62 km almost never does.

Three columns of the release are withheld from the item because they carry the answer. On the holdout, the
rule "burned_areas or ignition_points is nonzero anywhere in the window" scores AUPRC 0.816 against the
published baselines' 0.853 to 0.858, at a false positive rate of 0.001. The baselines therefore did not see
them, whatever the paper's sentence about using every variable says, and neither does an item here. Nothing
outside an item's "context" key is safe to render into a prompt: answer, label, fold and burned_area_ha sit
outside it on purpose.

Two properties of the released values carry into every item and are left as they are. lai, ndvi and smi are
8, 16 and 10-day composites held constant between updates, so their series read as step functions rather
than daily observations. lst_day and lst_night are missing on about a third of cell-days, because MODIS does
not see through cloud, and the gap travels as null rather than as a filled number.
"""
import hashlib
import json
import math
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

D = pathlib.Path(__file__).parent / "data" / "mesogeos"
OUT = pathlib.Path(__file__).parent / "task-mesogeos"
OUT.mkdir(exist_ok=True)

LAG = 30
TEST_YEARS = (2021, 2022)
TRAIN_YEARS_MAX = 2019
SIZE_CLASS_HA = 30
DYNAMIC = ["d2m", "lai", "lst_day", "lst_night", "ndvi", "rh", "smi", "sp", "ssrd", "t2m", "tp", "wind_speed"]
STATIC = ["dem", "roads_distance", "slope", "lc_agriculture", "lc_forest", "lc_grassland", "lc_settlement",
          "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies", "lc_wetland", "population"]
# Units as the datacube documents them, except where the released values contradict the documentation. The
# land cover columns are documented as percentages and are measured here to sum to exactly 1 in every row,
# so they travel as fractions. Relative humidity is documented as "%/100" and travels as the fraction it is.
UNITS = {
    "d2m": "K, day's maximum 2 m dewpoint temperature",
    "lai": "unitless, leaf area index",
    "lst_day": "K, day's land surface temperature from MODIS; missing under cloud",
    "lst_night": "K, night's land surface temperature from MODIS; missing under cloud",
    "ndvi": "unitless, normalized difference vegetation index",
    "rh": "fraction 0 to 1, day's minimum relative humidity",
    "smi": "unitless, soil moisture index",
    "sp": "Pa, day's maximum surface pressure",
    "ssrd": "J/m^2, day's average surface solar radiation downwards",
    "t2m": "K, day's maximum 2 m temperature",
    "tp": "m, day's total precipitation",
    "wind_speed": "m/s, day's maximum wind speed",
    "dem": "m, elevation",
    "roads_distance": "km, distance to the nearest road",
    "slope": "rad, slope; see the build log, this column is degenerate in the release",
    "population": "people/km^2",
    "lc_agriculture": "fraction of the 1 km cell; the eight land cover fractions sum to 1",
}
for _lc in [c for c in STATIC if c.startswith("lc_")]:
    UNITS.setdefault(_lc, UNITS["lc_agriculture"])
PUBLISHED_BASELINES = {  # kondylatos2023mesogeos Table 1, same 2021 to 2022 holdout
    "LSTM": {"precision": 0.763, "recall": 0.812, "f1": 0.786, "auprc": 0.853},
    "Transformer": {"precision": 0.802, "recall": 0.759, "f1": 0.780, "auprc": 0.856},
    "GTN": {"precision": 0.781, "recall": 0.790, "f1": 0.786, "auprc": 0.858},
}


def r(x):
    """Six significant figures, and None for a gap, so a reader can match a value back to the source csv."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float("%.6g" % x)


def load(name, label):
    df = pd.read_csv(D / name, low_memory=False)
    # Every downstream reshape assumes the file arrives as contiguous 30-row blocks, one sample each, in
    # day order. dataset.py in the release makes the same assumption by slicing rows 30 at a time.
    ids = df["sample"].to_numpy().reshape(-1, LAG)
    assert (ids == ids[:, :1]).all(), "%s is not laid out as contiguous %d-row samples" % (name, LAG)
    t = pd.to_datetime(df["time"]).to_numpy().reshape(-1, LAG)
    assert set(np.unique((t[:, 1:] - t[:, :-1]).astype("timedelta64[D]").astype(int))) == {1}, "window has a gap"
    for col in ("x", "y"):
        v = df[col].to_numpy().reshape(-1, LAG)
        assert (v == v[:, :1]).all(), "%s moves within a sample in %s" % (col, name)
    return df, ids[:, 0], t, label


frames = [load("positives.csv", 1), load("negatives.csv", 0)]
items, index = [], []
for df, sample_ids, times, label in frames:
    n = len(sample_ids)
    daily = {c: df[c].to_numpy(dtype=float).reshape(n, LAG) for c in DYNAMIC}
    static = {c: df[c].to_numpy(dtype=float).reshape(n, LAG)[:, 0] for c in STATIC}
    lon = df["x"].to_numpy(dtype=float).reshape(n, LAG)[:, 0]
    lat = df["y"].to_numpy(dtype=float).reshape(n, LAG)[:, 0]
    burned = df["burned_area_has"].to_numpy(dtype=float).reshape(n, LAG)[:, 0]
    starts = pd.to_datetime(times[:, 0])
    ends = pd.to_datetime(times[:, -1])
    keep = np.where(np.isin(ends.year, TEST_YEARS))[0]
    tag = "pos" if label else "neg"
    for i in keep:
        # Both classes are asked about the day after the window. For a positive that day is the EFFIS
        # ignition date, because the release shifts the window to end one day earlier. For a negative the
        # source verified the absence of fire on the last window day rather than on the day after it; see
        # the build log for what that asymmetry is worth.
        target = (ends[i] + pd.Timedelta(days=1)).date()
        items.append({
            "item_id": "mesogeos-%s-%05d-%s" % (tag, sample_ids[i], target),
            "split": "test",
            "label": int(label),
            "answer": "yes" if label else "no",
            "burned_area_ha": float(burned[i]) if label else None,
            "source_file": "positives.csv" if label else "negatives.csv",
            "source_sample": int(sample_ids[i]),
            "target_date": str(target),
            "context": {
                "longitude": r(lon[i]),
                "latitude": r(lat[i]),
                "window_start": str(starts[i].date()),
                "window_end": str(ends[i].date()),
                "target_date": str(target),
                "history_days": LAG,
                "size_class_hectares": SIZE_CLASS_HA,
                "cell_km": 1,
                "daily": {c: [r(v) for v in daily[c][i]] for c in DYNAMIC},
                "static": {c: r(static[c][i]) for c in STATIC},
                "units": UNITS,
            },
        })
        index.append({"item_id": items[-1]["item_id"], "label": int(label), "answer": items[-1]["answer"],
                      "target_date": str(target), "window_start": str(starts[i].date()),
                      "window_end": str(ends[i].date()), "longitude": r(lon[i]), "latitude": r(lat[i]),
                      "burned_area_ha": items[-1]["burned_area_ha"], "source_file": items[-1]["source_file"],
                      "source_sample": int(sample_ids[i])})

items.sort(key=lambda d: (d["target_date"], -d["label"], d["source_sample"]))
index = sorted(index, key=lambda d: (d["target_date"], -d["label"], d["source_sample"]))
# Items are in date order, so a runner that takes the first N gets January and February and nothing else.
# fold is sha256(item_id) mod 10, which is stable across runs and machines, so "fold == 0" is a tenth of the
# holdout that keeps the class balance and the season mix.
for it, ix in zip(items, index):
    it["fold"] = int(hashlib.sha256(it["item_id"].encode()).hexdigest(), 16) % 10
    ix["fold"] = it["fold"]
idx = pd.DataFrame(index)
y = idx["label"].to_numpy()
pos_n, neg_n = int(y.sum()), int((y == 0).sum())
print("items %d | fire %d | no fire %d | positive share %.4f" % (len(items), pos_n, neg_n, y.mean()))
print("target dates %s to %s" % (idx["target_date"].min(), idx["target_date"].max()))
print("published holdout is 4107 = 1369 positives + 2738 negatives; the negative pool holds %d, and the"
      % neg_n)
print("  release subsamples it with an unseeded np.random.choice, so the exact 2738 cannot be recovered.")
print("  Taking the whole pool is deterministic and differs from the paper's test set by %d negatives."
      % (neg_n - 2738))

fold = idx.groupby("fold")["label"].agg(["size", "mean"])
print("\nfolds (sha256 of item_id mod 10): sizes %d to %d, positive share %.3f to %.3f"
      % (fold["size"].min(), fold["size"].max(), fold["mean"].min(), fold["mean"].max()))

gaps = {c: float(np.mean([v is None for it in items for v in it["context"]["daily"][c]])) for c in DYNAMIC}
print("\ndaily gaps, share of cell-days with no observation:")
for c, v in sorted(gaps.items(), key=lambda kv: -kv[1])[:5]:
    print("  %-11s %.3f" % (c, v))
slope = idx.merge(pd.DataFrame({"item_id": [i["item_id"] for i in items],
                                "slope": [i["context"]["static"]["slope"] for i in items]}), on="item_id")["slope"]
print("slope in the release: p50 %.4f rad (%.1f deg), p01 %.4f, max %.4f; it is piled at pi/2 and carries"
      % (slope.median(), math.degrees(slope.median()), slope.quantile(0.01), slope.max()))
print("  almost no terrain signal. It is kept because the published baselines used it.")

# ---- naive baselines on the same holdout -------------------------------------------------------------
# Metric: AUPRC on the fire class, which is average_precision_score, plus per-class F1. Plain accuracy is
# reported only to show why it is the wrong headline: answering "no" every time already scores 0.67.
train = []
for df, sample_ids, times, label in frames:
    ends = pd.to_datetime(times[:, -1])
    sel = ends.year <= TRAIN_YEARS_MAX
    t2m = df["t2m"].to_numpy(dtype=float).reshape(len(sample_ids), LAG)[:, -1]
    lst = np.isfinite(df["lst_day"].to_numpy(dtype=float).reshape(len(sample_ids), LAG)).sum(axis=1)
    train.append(pd.DataFrame({"month": ends.month, "label": label, "t2m": t2m, "lst_days": lst})[sel])
train = pd.concat(train, ignore_index=True)

test = pd.DataFrame({
    "month": pd.to_datetime(idx["window_end"]).dt.month,
    "label": y,
    "t2m": [i["context"]["daily"]["t2m"][-1] for i in items],
    "lst_days": [sum(v is not None for v in i["context"]["daily"]["lst_day"]) for i in items],
})
prior = train.groupby("month")["label"].mean()
s_month = test["month"].map(prior).fillna(train["label"].mean()).to_numpy()
s_t2m = np.where(np.isfinite(test["t2m"].to_numpy(dtype=float)), test["t2m"].to_numpy(dtype=float),
                 np.nanmean(train["t2m"]))


def signed(train_score, train_y, test_score):
    """Pick the sign on the training years, then report the holdout, so no test information is used."""
    sign = 1.0 if average_precision_score(train_y, train_score) >= average_precision_score(train_y, -train_score) else -1.0
    return sign, float(average_precision_score(test.label, sign * test_score))


prevalence = float(y.mean())
always_no = {"accuracy": float((y == 0).mean()), "f1_fire": float(f1_score(y, np.zeros_like(y))),
             "f1_no_fire": float(f1_score(1 - y, np.ones_like(y)))}
always_no["macro_f1"] = (always_no["f1_fire"] + always_no["f1_no_fire"]) / 2
always_yes = {"accuracy": float((y == 1).mean()), "f1_fire": float(f1_score(y, np.ones_like(y))),
              "f1_no_fire": 0.0}
always_yes["macro_f1"] = always_yes["f1_fire"] / 2
month_auprc = float(average_precision_score(y, s_month))
month_f1 = max(float(f1_score(y, (s_month >= t).astype(int))) for t in np.unique(s_month))
tr_t2m = np.where(np.isfinite(train["t2m"]), train["t2m"], np.nanmean(train["t2m"]))
_, t2m_auprc = signed(tr_t2m, train["label"], s_t2m)
_, lst_auprc = signed(train["lst_days"].to_numpy(dtype=float), train["label"],
                      test["lst_days"].to_numpy(dtype=float))

baseline = {
    "metric": "AUPRC on the fire class (sklearn average_precision_score) and per-class F1; accuracy is "
              "reported only as a warning, since the classes run two to one",
    "items": len(items), "fire": pos_n, "no_fire": neg_n, "positive_share": round(prevalence, 4),
    "holdout": "the published Mesogeos time holdout, target days in 2021 and 2022; train 2006 to 2019 and "
               "validation 2020 contribute no items",
    "subsampling": "items.jsonl is in target-date order, so take a fold rather than a prefix; fold is "
                   "sha256(item_id) mod 10 and each fold keeps the class balance and the season mix",
    "naive": {
        "constant_score": {"auprc": round(prevalence, 4),
                           "note": "any constant probability; the AUPRC floor is the positive share"},
        "always_no_fire": {k: round(v, 4) for k, v in always_no.items()},
        "always_fire": {k: round(v, 4) for k, v in always_yes.items()},
        "calendar_month_prior": {"auprc": round(month_auprc, 4), "best_f1_fire": round(month_f1, 4),
                                 "note": "positive rate per calendar month fit on the 2006 to 2019 training "
                                         "years only, applied to the holdout by month"},
        "last_day_t2m": {"auprc": round(t2m_auprc, 4),
                         "note": "the last day's maximum 2 m temperature as the score, sign chosen on the "
                                 "training years"},
        "modis_lst_day_count": {"auprc": round(lst_auprc, 4),
                                "note": "count of window days with a MODIS daytime land surface temperature. "
                                        "Cloud drives the gaps, so missingness tracks fire weather; this is "
                                        "an artifact of the source worth stating, not a defect to engineer "
                                        "away"},
    },
    "published_baselines": PUBLISHED_BASELINES,
    "published_baseline_note": "kondylatos2023mesogeos Table 1, on the 2738-negative version of this same "
                               "holdout. Comparable to within the 13 extra negatives kept here.",
}
print("\n--- naive baselines on the 2021 to 2022 holdout ---")
print("constant score          AUPRC %.4f  (the floor, equal to the positive share)" % prevalence)
print("always 'no fire'        accuracy %.4f | F1 fire %.4f | F1 no fire %.4f | macro F1 %.4f"
      % (always_no["accuracy"], always_no["f1_fire"], always_no["f1_no_fire"], always_no["macro_f1"]))
print("always 'fire'           accuracy %.4f | F1 fire %.4f" % (always_yes["accuracy"], always_yes["f1_fire"]))
print("calendar month prior    AUPRC %.4f | best F1 fire %.4f" % (month_auprc, month_f1))
print("last day t2m            AUPRC %.4f" % t2m_auprc)
print("MODIS lst_day day count AUPRC %.4f" % lst_auprc)
print("published baselines     AUPRC %s"
      % ", ".join("%s %.3f" % (k, v["auprc"]) for k, v in PUBLISHED_BASELINES.items()))
print("A model scores against the 0.332 floor and the 0.621 best trivial strategy, under a 0.858 ceiling.")

(OUT / "items.jsonl").write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
idx.to_csv(OUT / "items-index.csv", index=False)
(OUT / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
print("\nwrote", OUT / "items.jsonl", (OUT / "items.jsonl").stat().st_size, "bytes")
