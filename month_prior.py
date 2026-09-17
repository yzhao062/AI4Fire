"""Calendar-month prior on the 386 scored Mesogeos items, fit on the 2006 to 2019 training years only.

The paper's earlier table row fitted the prior on the test items' own labels, which peeks. This is the
definition build_items_mesogeos.py uses for baseline.json: positive rate per window-end month over the
training samples, applied to each test item by its window-end month.
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

S = pathlib.Path(__file__).parent
D = S / "data" / "mesogeos"
LAG, TRAIN_YEARS_MAX = 30, 2019

parts = []
for name, label in (("positives.csv", 1), ("negatives.csv", 0)):
    df = pd.read_csv(D / name, usecols=["time"], low_memory=False)
    t = pd.to_datetime(df["time"]).to_numpy().reshape(-1, LAG)
    ends = pd.to_datetime(t[:, -1])
    parts.append(pd.DataFrame({"month": ends.month, "year": ends.year, "label": label}))
train = pd.concat(parts, ignore_index=True)
train = train[train["year"] <= TRAIN_YEARS_MAX]
prior = train.groupby("month")["label"].mean()
print("training samples %d, positive share %.4f, prior by month:" % (len(train), train["label"].mean()))
print(prior.round(3).to_dict())

items = [json.loads(l) for l in (S / "task-mesogeos" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
test = [i for i in items if i["split"] == "test" and i.get("fold", 0) == 0]
y = np.array([i["label"] for i in test])
m = np.array([int(i["context"]["window_end"][5:7]) for i in test])
s = np.array([prior.get(k, train["label"].mean()) for k in m])
print("scored items %d | AP of the training-fit calendar-month prior: %.4f" % (len(test), average_precision_score(y, s)))
m2 = np.array([pd.to_datetime(i["target_date"]).month for i in test])
s2 = np.array([prior.get(k, train["label"].mean()) for k in m2])
print("same prior keyed by target-date month: %.4f" % average_precision_score(y, s2))
