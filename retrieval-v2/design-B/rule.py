"""Family B: nearest-neighbour analogue retrieval for the AI4Fire allocation task.

Instead of drawing six rows at random from the v1 band, every variant ranks the eligible pool rows by a weighted
Euclidean distance over seven standardized features that are known on the report day, and takes the six nearest.
Ties are broken by analogue_id (ascending), so a draw never touches rng: the rule is deterministic without it.

Features (item side / pool side), in the order of the weight vectors below:
  0 pers    log(max(personnel_today, 1))                       / log(max(today, 1))
  1 acres   log1p(max(acres, 0))                               / log1p(max(acres, 0))
  2 pct     percent_contained clipped to 0..100                / pct clipped to 0..100
  3 day     log(day_of_run)                                    / log(day_of_run)
  4 change  log(personnel_today / personnel_last_days[-2])     / log(max(today, 1) / max(prev, 1))
  5 new     log1p(max(new_acres, 0))                           / log1p(max(new_acres, 0))
  6 aerial  log1p(max(aerial_resources, 0))                    / log1p(max(aerial, 0))
Each feature is standardized as (x - mean) / std with the mean and std fitted once on pool rows dated before
2015-01-01 (fit_scales.py) and frozen in scales.json beside this file. A missing feature on the pool side is set
to the fitted mean (z = 0); a missing feature on the item side drops that feature from the distance for that item.

Scopes: "band" restricts the ranking to the v1 band (same acres, containment, and personnel band, exactly the key
v1 uses); "pers" restricts it to the same personnel band only; "all" ranks every eligible row.
Optional cap: at most `cap` rows from any one incident, taken in distance order.

Interface: RULES = {name: rule}, rule(cands, item, rng) -> list of pool rows (at most K). FROZEN names the
submitted variant. Call prepare(pool) once after building the pool for speed; the rule registers unseen rows on
its own otherwise, so it also works without that call.
"""
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
K = 6
ACRE_EDGES = [100, 1000, 5000, 20000, 100000]
PCT_EDGES = [10, 30, 60, 90]
PERS_EDGES = [25, 75, 200, 500]
FEATURES = ["pers", "acres", "pct", "day", "change", "new", "aerial"]
FIT_BEFORE = "2015-01-01"

# ----------------------------------------------------------------------------------------------------------------
# frozen scales (fitted by fit_scales.py on pool rows dated before 2015-01-01)
# ----------------------------------------------------------------------------------------------------------------
SCALES = json.loads((HERE / "scales.json").read_text(encoding="utf-8"))
MEAN = np.array([SCALES["mean"][f] for f in FEATURES], dtype=float)
STD = np.array([SCALES["std"][f] for f in FEATURES], dtype=float)


def band(value, edges):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def key_of(acres, pct, pers):
    return (band(acres, ACRE_EDGES), band(pct, PCT_EDGES), band(pers, PERS_EDGES))


def _isnum(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def raw_row(r):
    """Raw (unstandardized) feature vector of a pool row; NaN where the row lacks the field."""
    return [
        math.log(max(r["today"], 1.0)),
        math.log1p(max(r["acres"], 0.0)) if _isnum(r["acres"]) else math.nan,
        min(max(float(r["pct"]), 0.0), 100.0) if _isnum(r["pct"]) else math.nan,
        math.log(max(r["day_of_run"], 1)),
        math.log(max(r["today"], 1.0) / max(r["prev"], 1.0)) if _isnum(r["prev"]) else math.nan,
        math.log1p(max(r["new_acres"], 0.0)) if _isnum(r["new_acres"]) else math.nan,
        math.log1p(max(r["aerial"], 0.0)) if _isnum(r["aerial"]) else math.nan,
    ]


def raw_item(item):
    """Raw feature vector of an item from its context and day_of_run; NaN where the report lacks the field."""
    c = item["context"]
    hist = c.get("personnel_last_days") or []
    today = float(c["personnel_today"])
    change = math.log(max(today, 1.0) / max(float(hist[-2]), 1.0)) if len(hist) >= 2 and _isnum(hist[-2]) else math.nan
    return [
        math.log(max(today, 1.0)),
        math.log1p(max(c["acres"], 0.0)) if _isnum(c["acres"]) else math.nan,
        min(max(float(c["percent_contained"]), 0.0), 100.0) if _isnum(c["percent_contained"]) else math.nan,
        math.log(max(int(item["day_of_run"]), 1)),
        change,
        math.log1p(max(c["new_acres"], 0.0)) if _isnum(c["new_acres"]) else math.nan,
        math.log1p(max(c["aerial_resources"], 0.0)) if _isnum(c["aerial_resources"]) else math.nan,
    ]


def _keycode(key):
    """v1 band key -> small integer (an unknown band becomes 6, which no numeric band uses)."""
    a, p, s = key
    a = 6 if a == "unknown" else a
    p = 6 if p == "unknown" else p
    s = 6 if s == "unknown" else s
    return (a * 7 + p) * 7 + s


# ----------------------------------------------------------------------------------------------------------------
# registry of pool rows: standardized features, band codes, incident codes, id ranks
# ----------------------------------------------------------------------------------------------------------------
class _Registry:
    def __init__(self):
        self.index = {}          # analogue_id -> row position
        self.ids = []            # analogue_id per position
        self.z = np.zeros((0, len(FEATURES)))
        self.keycode = np.zeros(0, dtype=np.int64)
        self.persband = np.zeros(0, dtype=np.int64)
        self.incident = np.zeros(0, dtype=np.int64)
        self.idrank = np.zeros(0, dtype=np.int64)
        self._inc_codes = {}

    def register(self, rows):
        new = [r for r in rows if r["analogue_id"] not in self.index]
        if not new:
            return
        raw = np.array([raw_row(r) for r in new], dtype=float)
        z = (raw - MEAN) / STD
        z[np.isnan(z)] = 0.0                       # a pool row missing a feature sits at the fitted mean
        kc = np.array([_keycode(r["key"]) for r in new], dtype=np.int64)
        pb = np.array([band(r["today"], PERS_EDGES) for r in new], dtype=np.int64)
        inc = np.array([self._inc_codes.setdefault(r["incident_id"], len(self._inc_codes)) for r in new], dtype=np.int64)
        base = len(self.ids)
        for j, r in enumerate(new):
            self.index[r["analogue_id"]] = base + j
            self.ids.append(r["analogue_id"])
        self.z = np.vstack([self.z, z])
        self.keycode = np.concatenate([self.keycode, kc])
        self.persband = np.concatenate([self.persband, pb])
        self.incident = np.concatenate([self.incident, inc])
        order = sorted(range(len(self.ids)), key=lambda i: self.ids[i])
        self.idrank = np.empty(len(self.ids), dtype=np.int64)
        self.idrank[np.array(order, dtype=np.int64)] = np.arange(len(self.ids))

    def positions(self, cands):
        idx = self.index
        try:
            return np.fromiter((idx[r["analogue_id"]] for r in cands), dtype=np.int64, count=len(cands))
        except KeyError:
            self.register(cands)
            return np.fromiter((idx[r["analogue_id"]] for r in cands), dtype=np.int64, count=len(cands))


REG = _Registry()


def prepare(pool):
    """Standardize every pool row once. Optional: the rule registers rows it has not seen on its own."""
    REG.register(pool)


# ----------------------------------------------------------------------------------------------------------------
# the rule factory
# ----------------------------------------------------------------------------------------------------------------
WEIGHTS = {
    # order: pers, acres, pct, day, change, new, aerial
    "eq":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # every feature weighs the same
    "chg": [1.0, 1.0, 1.0, 1.0, 4.0, 3.0, 1.0],   # the recent-change features weigh heavily
    "dyn": [1.0, 1.0, 1.0, 3.0, 4.0, 3.0, 1.0],   # change features and day of run weigh heavily
    "sta": [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],   # the state features (the v1 band's three) weigh heavily
    # ablations, to say what does the work
    "nochg": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],     # eq without the recent personnel change
    "band3": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],     # nearest on the v1 band's three variables only
}


def item_vector(item, weights):
    """Standardized item vector and the effective weights (zero where the item lacks the feature)."""
    w = np.array(WEIGHTS[weights] if isinstance(weights, str) else weights, dtype=float)
    zi = (np.array(raw_item(item), dtype=float) - MEAN) / STD
    w = np.where(np.isnan(zi), 0.0, w)
    return np.nan_to_num(zi), w


def make_rule(scope, weights, cap=None):
    if scope not in ("band", "pers", "all"):
        raise ValueError(scope)

    def rule(cands, item, rng):
        # rng is not used: the six nearest rows are a deterministic function of the item and its candidate set
        if not cands:
            return []
        pos = REG.positions(cands)
        zi, w = item_vector(item, weights)
        c = item["context"]
        if scope == "band":
            mask = REG.keycode[pos] == _keycode(key_of(c["acres"], c["percent_contained"], c["personnel_today"]))
        elif scope == "pers":
            mask = REG.persband[pos] == band(c["personnel_today"], PERS_EDGES)
        else:
            mask = np.ones(len(pos), dtype=bool)
        where = np.nonzero(mask)[0]                # positions into cands
        if len(where) == 0:
            return []
        sub = pos[where]
        d = ((REG.z[sub] - zi) ** 2 * w).sum(axis=1)
        order = np.lexsort((REG.idrank[sub], d))   # by distance, then analogue_id
        if cap is None:
            take = order[:K]
        else:
            take, seen = [], {}
            for j in order:
                inc = int(REG.incident[sub[j]])
                if seen.get(inc, 0) >= cap:
                    continue
                seen[inc] = seen.get(inc, 0) + 1
                take.append(j)
                if len(take) == K:
                    break
        return [cands[int(where[j])] for j in take]

    rule.__name__ = "nn_%s_%s%s" % (scope, weights if isinstance(weights, str) else "custom",
                                    "" if cap is None else "_cap%d" % cap)
    rule.scope, rule.weights, rule.cap = scope, weights, cap
    return rule


RULES = {}
for _scope in ("band", "pers", "all"):
    for _w in ("eq", "chg", "dyn", "sta"):
        RULES["nn_%s_%s" % (_scope, _w)] = make_rule(_scope, _w)
for _scope in ("pers", "all"):
    for _w in ("chg", "dyn"):
        RULES["nn_%s_%s_cap2" % (_scope, _w)] = make_rule(_scope, _w, cap=2)
for _scope in ("pers", "all"):
    for _w in ("nochg", "band3"):
        RULES["nn_%s_%s" % (_scope, _w)] = make_rule(_scope, _w)

# The frozen choice, by the brief's prespecified rule applied within this family on the development table
# (report.md): nearest six within the personnel band, change-heavy weights, no per-incident cap, no rng.
FROZEN = "nn_pers_chg"
