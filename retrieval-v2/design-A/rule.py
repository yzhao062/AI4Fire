"""Family A: recent-dynamics matching for the allocation analogue-retrieval rule.

Every variant keeps v1's band on acres, containment, and personnel and adds, in a ladder, conditions on the
dynamics visible on the report day. The components a variant may match on:
  band   v1's (acres band, containment band, personnel band); always first.
  p3     the one-day personnel change, today against the day before (item: personnel_last_days[-2] ->
         personnel_today; pool row: prev -> today), three ways: fell by more than a tenth, held within a tenth,
         rose by more than a tenth.
  p5     the same change five ways: fell by more than a quarter, fell by a tenth to a quarter, held, rose by a
         tenth to a quarter, rose by more than a quarter.
  p4x    exact hold (count unchanged), small change within a tenth, fell, rose.
  p6x    p4x with fell and rose each split at a quarter.
  p5s    p4x with the exact hold split by whether it has lasted two days (item: personnel_last_days[-3:] all
         equal; pool row: prev == today and the same incident's row for the day before also shows prev == today,
         looked up among the candidate rows the harness already filtered, so nothing dated on or after the
         report day is touched).
  p7s    p6x with the same two-day split of the exact hold.
  t3     the two-day personnel change, today against two days before, three ways as p3.
  a      whether acres grew over the last day (item: acres_last_days[-2] -> [-1], falling back to new_acres > 0;
         pool row: acres_prev -> acres, falling back to new_acres > 0).
Containment change (pct against pct_prev) exists on pool rows but the item context carries no containment
history, only today's percent contained, so it cannot be matched and is not used.

A variant is a tuple of components. Ladder: match on all of them; if fewer than MIN_ROWS rows match, drop the last
component and try again, down to the v1 band alone. Because p5 refines p3 (and p6x refines p5, p7s refines p6x),
a tuple such as (band, p3, p5) relaxes by coarsening the personnel-change bucket before dropping it. Six rows are
then sampled at random with the item rng, exactly as v1 does. The "fill" variants instead take every row at the
most specific level that has any and top up from the next levels until six are drawn.

Interface: RULES = {name: fn}, fn(cands, item, rng) -> list of pool rows; FROZEN names the submitted variant.
There is no fitted component: the bucket edges are the prespecified constants below (a tenth, the harness's own
definition of a moving day, and a quarter for the finer split).

Pool rows are dict objects shared across calls; the rule caches derived features on them under keys that start
with an underscore ("_dyn", "_prev2", "_kA:<components>"). The harness reads none of those keys.
"""
import pandas as pd

from dev_eval import K, key_of

ONE_DAY = pd.Timedelta(days=1)
HOLD = 0.10    # |today/prev - 1| <= HOLD counts as held, the harness's own movement threshold
BIG = 0.25     # the finer split separates changes beyond a quarter


# ----------------------------------------------------------------------------------------------- raw features
def _ratio(today, prev):
    if prev is None or today is None:
        return None
    return today / max(prev, 1)


def _grew(acres, acres_prev, new_acres):
    """True / False / None: did acres grow over the last day."""
    if acres is not None and acres_prev is not None:
        return acres > acres_prev
    if new_acres is not None:
        return new_acres > 0
    return None


def _ensure_prev2(cands):
    """Cache, on each candidate row, the personnel count two days before (from the same incident's earlier row)."""
    missing = [r for r in cands if "_prev2" not in r]
    if not missing:
        return
    idx = {(r["incident_id"], r["date"]): r for r in cands}
    for r in missing:
        p = None
        if r["prev"] is not None:
            q = idx.get((r["incident_id"], r["date"] - ONE_DAY))
            if q is not None:
                p = q["prev"]
        r["_prev2"] = p


def _row_dyn(r, with2):
    """Dynamics of a pool row: (ratio prev->today, exact hold, held two days, ratio prev2->today, acres grew).

    The one-day part is cached on the row under "_dyn"; the two-day part under "_dyn2", and only when a two-day
    component asks for it (with2), after _ensure_prev2 has run on the candidate rows. A variant without a two-day
    component sees placeholders there and never reads them.
    """
    d = r.get("_dyn")
    if d is None:
        d = r["_dyn"] = (_ratio(r["today"], r["prev"]), r["prev"] is not None and r["today"] == r["prev"],
                         _grew(r["acres"], r["acres_prev"], r["new_acres"]))
    if not with2:
        return (d[0], d[1], False, None, d[2])
    e = r.get("_dyn2")
    if e is None:
        prev2 = r["_prev2"]
        e = r["_dyn2"] = (bool(d[1] and prev2 is not None and prev2 == r["today"]), _ratio(r["today"], prev2))
    return (d[0], d[1], e[0], e[1], d[2])


def _item_dyn(item):
    """The same five dynamics from the item's context only."""
    c = item["context"]
    pl = c.get("personnel_last_days") or []
    today = c["personnel_today"]
    prev = pl[-2] if len(pl) >= 2 else None
    prev2 = pl[-3] if len(pl) >= 3 else None
    ratio = _ratio(today, prev)
    exact = prev is not None and today == prev
    hold2 = bool(exact and prev2 is not None and prev2 == today)
    ratio2 = _ratio(today, prev2)
    al = c.get("acres_last_days") or []
    acres_prev = al[-2] if len(al) >= 2 else None
    return (ratio, exact, hold2, ratio2, _grew(c.get("acres"), acres_prev, c.get("new_acres")))


# ----------------------------------------------------------------------------------------------- bucketings
def _three(ratio):
    if ratio is None:
        return "u"
    if ratio < 1 - HOLD:
        return "f"
    if ratio > 1 + HOLD:
        return "r"
    return "h"


def _five(ratio):
    if ratio is None:
        return "u"
    if ratio < 1 - BIG:
        return "F"
    if ratio < 1 - HOLD:
        return "f"
    if ratio > 1 + BIG:
        return "R"
    if ratio > 1 + HOLD:
        return "r"
    return "h"


def _p3(d):
    return _three(d[0])


def _p5(d):
    return _five(d[0])


def _p4x(d):
    ratio, exact = d[0], d[1]
    if ratio is None:
        return "u"
    if exact:
        return "e"
    b = _three(ratio)
    return "s" if b == "h" else b


def _p6x(d):
    ratio, exact = d[0], d[1]
    if ratio is None:
        return "u"
    if exact:
        return "e"
    b = _five(ratio)
    return "s" if b == "h" else b


def _p5s(d):
    b = _p4x(d)
    return "ee" if b == "e" and d[2] else b


def _p7s(d):
    b = _p6x(d)
    return "ee" if b == "e" and d[2] else b


def _t3(d):
    return _three(d[3])


def _t5(d):
    return _five(d[3])


def _a(d):
    return d[4]


COMPONENTS = {"p3": _p3, "p5": _p5, "p4x": _p4x, "p6x": _p6x, "p5s": _p5s, "p7s": _p7s, "t3": _t3, "t5": _t5, "a": _a}
NEEDS_PREV2 = {"p5s", "p7s", "t3", "t5"}


# ----------------------------------------------------------------------------------------------- rule factory
def make_rule(components, min_rows=3, fill=False):
    """A ladder rule over the given components; "band" must come first; the ladder relaxes from the end.

    Each call groups the candidate rows once by their full key (cached on the row under "_kA:<components>"); a
    ladder level then collects the groups whose prefix matches the item's key, in first-appearance order, so the
    draw is deterministic given the candidate order the harness passes.
    """
    assert components[0] == "band" and all(c in COMPONENTS for c in components[1:])
    fns = [COMPONENTS[c] for c in components[1:]]
    ck = "_kA:" + "+".join(components)
    depth = len(components)
    needs_prev2 = bool(NEEDS_PREV2 & set(components))

    def row_key(r):
        k = r.get(ck)
        if k is None:
            d = _row_dyn(r, needs_prev2)
            k = r[ck] = (r["key"],) + tuple(f(d) for f in fns)
        return k

    def rule(cands, item, rng):
        if needs_prev2:
            _ensure_prev2(cands)
        c = item["context"]
        d = _item_dyn(item)
        ik = (key_of(c["acres"], c["percent_contained"], c["personnel_today"]),) + tuple(f(d) for f in fns)
        groups = {}
        for r in cands:
            k = row_key(r)
            g = groups.get(k)
            if g is None:
                groups[k] = [r]
            else:
                g.append(r)
        if not fill:
            hits = []
            for n in range(depth, 0, -1):        # most specific first, the v1 band alone last
                pre = ik[:n]
                hits = [r for k, g in groups.items() if k[:n] == pre for r in g]
                if len(hits) >= min_rows:
                    break
            return rng.sample(hits, min(K, len(hits)))
        chosen, seen = [], set()
        for n in range(depth, 0, -1):
            pre = ik[:n]
            hits = [r for k, g in groups.items() if k[:n] == pre for r in g if id(r) not in seen]
            need = K - len(chosen)
            take = hits if len(hits) <= need else rng.sample(hits, need)
            chosen.extend(take)
            seen.update(id(r) for r in take)
            if len(chosen) >= K:
                break
        return chosen

    rule.__name__ = "rule_" + "_".join(components) + "_min%d%s" % (min_rows, "_fill" if fill else "")
    rule.components, rule.min_rows, rule.fill = tuple(components), min_rows, fill
    return rule


RULES = {
    # round 1: one personnel-change bucketing, then the v1 band
    "A-b3-p":        make_rule(("band", "p3")),
    "A-b5-p":        make_rule(("band", "p5")),
    "A-b4x-p":       make_rule(("band", "p4x")),
    "A-b6x-p":       make_rule(("band", "p6x")),
    # round 1: personnel change, then acres growth (relaxed first), then the v1 band
    "A-b3-pa":       make_rule(("band", "p3", "a")),
    "A-b5-pa":       make_rule(("band", "p5", "a")),
    "A-b4x-pa":      make_rule(("band", "p4x", "a")),
    "A-b6x-pa":      make_rule(("band", "p6x", "a")),
    # round 1: acres growth added first, personnel change relaxed first (control for the ordering)
    "A-b4x-ap":      make_rule(("band", "a", "p4x")),
    # round 1: two-day hold streak split off from the one-day exact hold
    "A-b5s-p":       make_rule(("band", "p5s")),
    "A-b5s-pa":      make_rule(("band", "p5s", "a")),
    "A-b7s-p":       make_rule(("band", "p7s")),
    "A-b7s-pa":      make_rule(("band", "p7s", "a")),
    # round 1: ladder threshold and fill variants on the streak bucketing
    "A-b5s-pa-min6": make_rule(("band", "p5s", "a"), min_rows=6),
    "A-b5s-pa-fill": make_rule(("band", "p5s", "a"), fill=True),
    "A-b5s-p-min6":  make_rule(("band", "p5s"), min_rows=6),
    "A-b5s-p-fill":  make_rule(("band", "p5s"), fill=True),
    # round 2: coarsening ladders (five-way relaxes to three-way before the band)
    "A-p3>5":        make_rule(("band", "p3", "p5")),
    "A-p3>5>6x":     make_rule(("band", "p3", "p5", "p6x")),
    "A-p3>5>6x>7s":  make_rule(("band", "p3", "p5", "p6x", "p7s")),
    # round 2: the two-day trend as a further level
    "A-b5-p-t3":     make_rule(("band", "p5", "t3")),
    "A-p3>5-t3":     make_rule(("band", "p3", "p5", "t3")),
    "A-b3-p-t3":     make_rule(("band", "p3", "t3")),
    # round 2: fill and min6 on the five-way ladders
    "A-b5-p-fill":   make_rule(("band", "p5"), fill=True),
    "A-b5-p-min6":   make_rule(("band", "p5"), min_rows=6),
    "A-p3>5-fill":   make_rule(("band", "p3", "p5"), fill=True),
    "A-p3>5-min6":   make_rule(("band", "p3", "p5"), min_rows=6),
    # round 3: refinements of the one-day-then-two-day ladder
    "A-b5-p-t5":       make_rule(("band", "p5", "t5")),
    "A-b5-t3-p":       make_rule(("band", "t3", "p5")),
    "A-b6x-p-t3":      make_rule(("band", "p6x", "t3")),
    "A-b5-p-t3-a":     make_rule(("band", "p5", "t3", "a")),
    "A-b5-p-t3-fill":  make_rule(("band", "p5", "t3"), fill=True),
    "A-b5-p-t3-min6":  make_rule(("band", "p5", "t3"), min_rows=6),
    # round 4: the two-day trend on its own, and thresholds for the trend-first ordering
    "A-t3":            make_rule(("band", "t3")),
    "A-t5":            make_rule(("band", "t5")),
    "A-t3-p3":         make_rule(("band", "t3", "p3")),
    "A-t5-p5":         make_rule(("band", "t5", "p5")),
    "A-b5-t3-p-fill":  make_rule(("band", "t3", "p5"), fill=True),
    "A-b5-t3-p-min6":  make_rule(("band", "t3", "p5"), min_rows=6),
}

FROZEN = "A-b5-t3-p"   # chosen by the prespecified selection rule on the development run; see report.md
