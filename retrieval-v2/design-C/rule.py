"""Family C: stability-aware stratified analogue draw.

The paper's diagnosis is that a set matched on acres, containment, and staffing is not matched on the
probability that the count moves at all. This family estimates that probability from the item's own report
(p_hat), then draws the six analogues so that the drawn set carries it: n_move = round(6 * p_hat) rows whose
next-day ratio moved by more than a tenth, 6 - n_move rows that held, both from the v1 band and topped up from
the personnel band when a stratum is thin. The analogue-only prediction (persistence times the median drawn
ratio) then moves only when the item's context says movement is more likely than not, and the displayed share
of moving analogues is the estimate of P(move) itself.

Variants (RULES):
  C-model         p_hat from a logistic model (model.json), moved rows drawn regardless of direction
  C-model-trend   as C-model, moved rows matched to the direction of the item's most recent change
  C-model-dir     as C-model, moved rows split into up and down by a second logistic P(up | move)
  C-phase         p_hat from a (day-of-run bucket x containment band) table, no direction control
  C-phase-trend   phase prior plus trend-matched direction
  C-phase-dir     phase prior plus the fitted direction split
  C-hgb           p_hat from a HistGradientBoostingClassifier (hgb.pkl), no direction control
  C-hgb-dir       HGB movement probability plus the fitted direction split
All constants come from model.json (and hgb.pkl for the two HGB variants), written by fit.py from pool rows
dated before 2015-01-01 only. Every variant uses only the item's context and day_of_run plus the candidate rows
the harness passes. Determinism: only the item rng is used.

Rule interface: rule(cands, item, rng) -> list of at most K pool rows (see dev_eval.py).
"""
import json
import math
import pathlib
import pickle

HERE = pathlib.Path(__file__).resolve().parent
K = 6
MOVE = 0.1  # a ratio moves when |ratio - 1| > MOVE, the harness's definition
ACRE_EDGES = [100, 1000, 5000, 20000, 100000]
PCT_EDGES = [10, 30, 60, 90]
PERS_EDGES = [25, 75, 200, 500]
DOR_EDGES = [2, 3, 4, 6, 8, 11, 15, 21, 31]  # day-of-run buckets for the phase prior: 1, 2, 3, 4-5, 6-7, 8-10, 11-14, 15-20, 21-30, 31+
RC_CLIP = 1.5

LIN_FEATURES = ["log_personnel", "log_day_of_run", "recent_change", "abs_recent_change", "flat_yesterday",
                "log1p_new_acres", "new_acres_missing", "log1p_acres", "acres_missing", "log1p_aerial", "aerial_missing",
                "pct_band_0", "pct_band_1", "pct_band_2", "pct_band_3", "pct_band_4", "pct_missing"]
HGB_FEATURES = ["log_personnel", "pct", "log_day_of_run", "recent_change", "abs_recent_change", "log1p_new_acres",
                "log1p_acres", "log1p_aerial"]


# ----------------------------------------------------------------------------------------------- bands (as v1)
def band(value, edges):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def key_of(acres, pct, pers):
    return (band(acres, ACRE_EDGES), band(pct, PCT_EDGES), band(pers, PERS_EDGES))


def row_key(r):
    k = r.get("key")
    return k if k is not None else key_of(r["acres"], r["pct"], r["today"])


def ratio(r):
    return r["next"] / max(r["today"], 1)


def _isnum(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ----------------------------------------------------------------------------------------------- features
def lin_features(today, prev, pct, day_of_run, new_acres, acres, aerial):
    """Feature vector for the two logistic models, from report-day quantities only."""
    lp = math.log(max(today, 1.0))
    rc = 0.0
    if _isnum(prev):
        rc = max(-RC_CLIP, min(RC_CLIP, math.log(max(today, 1.0) / max(prev, 1.0))))
    pb = band(pct, PCT_EDGES)
    return [
        lp, math.log(max(day_of_run, 1)), rc, abs(rc), 1.0 if rc == 0.0 else 0.0,
        math.log1p(new_acres) if _isnum(new_acres) and new_acres >= 0 else 0.0, 0.0 if _isnum(new_acres) else 1.0,
        math.log1p(acres) if _isnum(acres) and acres >= 0 else 0.0, 0.0 if _isnum(acres) else 1.0,
        math.log1p(aerial) if _isnum(aerial) and aerial >= 0 else 0.0, 0.0 if _isnum(aerial) else 1.0,
        1.0 if pb == 0 else 0.0, 1.0 if pb == 1 else 0.0, 1.0 if pb == 2 else 0.0, 1.0 if pb == 3 else 0.0,
        1.0 if pb == 4 else 0.0, 1.0 if pb == "unknown" else 0.0,
    ]


def hgb_features(today, prev, pct, day_of_run, new_acres, acres, aerial):
    """Compact vector with NaN for missing values, for the gradient-boosting model."""
    nan = float("nan")
    lp = math.log(max(today, 1.0))
    rc = nan
    if _isnum(prev):
        rc = max(-RC_CLIP, min(RC_CLIP, math.log(max(today, 1.0) / max(prev, 1.0))))
    return [lp, pct if _isnum(pct) else nan, math.log(max(day_of_run, 1)), rc, abs(rc) if rc == rc else nan,
            math.log1p(new_acres) if _isnum(new_acres) and new_acres >= 0 else nan,
            math.log1p(acres) if _isnum(acres) and acres >= 0 else nan,
            math.log1p(aerial) if _isnum(aerial) and aerial >= 0 else nan]


def row_quantities(r):
    """(today, prev, pct, day_of_run, new_acres, acres, aerial) of a pool row."""
    return (r["today"], r["prev"], r["pct"], r["day_of_run"], r["new_acres"], r["acres"], r["aerial"])


def item_quantities(item):
    """The same seven quantities from an item's context: everything is on the report day or before."""
    c = item["context"]
    last = c.get("personnel_last_days") or []
    prev = last[-2] if len(last) >= 2 else None  # the list ends with today's count
    return (c["personnel_today"], prev, c["percent_contained"], item["day_of_run"], c["new_acres"], c["acres"],
            c["aerial_resources"])


def item_trend(item):
    """Sign of the most recent change: +1 up, -1 down, 0 flat or unknown."""
    today, prev = item_quantities(item)[:2]
    if not _isnum(prev):
        return 0
    return (today > prev) - (today < prev)


# ----------------------------------------------------------------------------------------------- frozen models
_MODEL = None
_HGB = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
    return _MODEL


def hgb():
    global _HGB
    if _HGB is None:
        with open(HERE / "hgb.pkl", "rb") as f:
            _HGB = pickle.load(f)
    return _HGB


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _linear(spec, x):
    return spec["intercept"] + sum(w * v for w, v in zip(spec["weights"], x))


def p_move_model(item):
    return _sigmoid(_linear(model()["move"], lin_features(*item_quantities(item))))


def p_up_model(item):
    return _sigmoid(_linear(model()["up"], lin_features(*item_quantities(item))))


def p_move_phase(item):
    m = model()["phase"]
    q = item_quantities(item)
    cell = "%s|%s" % (band(q[3], DOR_EDGES), band(q[2], PCT_EDGES))
    return m["table"].get(cell, m["global"])


def p_move_hgb(item):
    import numpy as np
    x = np.array([hgb_features(*item_quantities(item))], dtype=float)
    return float(hgb()["move"].predict_proba(x)[0, 1])


# ----------------------------------------------------------------------------------------------- the draw
def _round_half_up(x):
    return int(math.floor(x + 0.5))


def _take(primary, secondary, n, rng):
    """n rows from primary; when primary is thin, all of it plus a top-up from secondary."""
    if n <= 0:
        return []
    if len(primary) >= n:
        return rng.sample(primary, n)
    return list(primary) + rng.sample(secondary, min(n - len(primary), len(secondary)))


def draw(cands, item, rng, p_move, direction="mixed", p_up=None):
    """Stratified draw: round(K * p_move) moved rows and the rest held rows, v1 band first, personnel band as top-up.

    direction: "mixed"  moved rows from either direction,
               "trend"  moved rows in the direction of the item's latest change (mixed when flat),
               "model"  moved rows split round(n_move * p_up) up and the rest down.
    """
    c = item["context"]
    key = key_of(c["acres"], c["percent_contained"], c["personnel_today"])
    pers = key[2]
    prim = {"held": [], "up": [], "down": []}
    seco = {"held": [], "up": [], "down": []}
    for r in cands:
        k = row_key(r)
        if k[2] != pers:
            continue
        x = ratio(r)
        s = "held" if abs(x - 1) <= MOVE else ("up" if x > 1 else "down")
        (prim if k == key else seco)[s].append(r)
    n_move = max(0, min(K, _round_half_up(K * p_move)))
    n_held = K - n_move
    rows = _take(prim["held"], seco["held"], n_held, rng)
    if direction == "model":
        n_up = max(0, min(n_move, _round_half_up(n_move * p_up)))
        rows += _take(prim["up"], seco["up"], n_up, rng)
        rows += _take(prim["down"], seco["down"], n_move - n_up, rng)
    else:
        t = item_trend(item) if direction == "trend" else 0
        if t > 0:
            rows += _take(prim["up"], seco["up"], n_move, rng)
        elif t < 0:
            rows += _take(prim["down"], seco["down"], n_move, rng)
        else:
            rows += _take(prim["up"] + prim["down"], seco["up"] + seco["down"], n_move, rng)
    rng.shuffle(rows)
    return rows


def rule_model(cands, item, rng):
    return draw(cands, item, rng, p_move_model(item), "mixed")


def rule_model_trend(cands, item, rng):
    return draw(cands, item, rng, p_move_model(item), "trend")


def rule_model_dir(cands, item, rng):
    return draw(cands, item, rng, p_move_model(item), "model", p_up_model(item))


def rule_phase(cands, item, rng):
    return draw(cands, item, rng, p_move_phase(item), "mixed")


def rule_phase_trend(cands, item, rng):
    return draw(cands, item, rng, p_move_phase(item), "trend")


def rule_phase_dir(cands, item, rng):
    return draw(cands, item, rng, p_move_phase(item), "model", p_up_model(item))


def rule_hgb(cands, item, rng):
    return draw(cands, item, rng, p_move_hgb(item), "mixed")


def rule_hgb_dir(cands, item, rng):
    return draw(cands, item, rng, p_move_hgb(item), "model", p_up_model(item))


RULES = {
    "C-model": rule_model,
    "C-model-trend": rule_model_trend,
    "C-model-dir": rule_model_dir,
    "C-phase": rule_phase,
    "C-phase-trend": rule_phase_trend,
    "C-phase-dir": rule_phase_dir,
    "C-hgb": rule_hgb,
    "C-hgb-dir": rule_hgb_dir,
}
FROZEN = "C-model-dir"  # set after the development run; see report.md
