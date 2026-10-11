"""Train the AgroFarm models and export them for the web app.

    pip install -r ml/requirements.txt
    python ml/train.py

Writes:
  lib/model/crop.json        random forest (22 crops) + per-crop growing profiles
  lib/model/fertilizer.json  decision tree (7 fertilizers)
  ml/metrics.json            held-out / cross-validated scores shown on the site
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ml" / "data"
OUT = ROOT / "lib" / "model"
OUT.mkdir(parents=True, exist_ok=True)

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def export_tree(est, n_classes):
    """Flatten a fitted sklearn tree. Leaves carry sparse class fractions."""
    t = est.tree_
    leaf = t.children_left == -1
    nodes = {
        "f": [int(f) if not lf else -1 for f, lf in zip(t.feature, leaf)],
        "t": [float(x) if not lf else 0 for x, lf in zip(t.threshold, leaf)],
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "v": [],
    }
    for i in range(t.node_count):
        if not leaf[i]:
            nodes["v"].append(None)
            continue
        dist = t.value[i][0] / t.value[i][0].sum()
        nodes["v"].append([[int(c), round(float(p), 5)] for c, p in enumerate(dist) if p > 0])
    return nodes


def predict_exported(trees, x, n_classes):
    """Mirror of lib/forest.ts, used to check parity with sklearn."""
    proba = np.zeros(n_classes)
    x = np.asarray(x, dtype=np.float32).astype(np.float64)  # sklearn compares float32 inputs in float64
    for tr in trees:
        i = 0
        while tr["f"][i] != -1:
            i = tr["l"][i] if x[tr["f"][i]] <= tr["t"][i] else tr["r"][i]
        for c, p in tr["v"][i]:
            proba[c] += p
    return proba / len(trees)


# ---------------------------------------------------------------- crop model
crop = pd.read_csv(DATA / "crop_recommendation.csv")
X, y = crop[CROP_FEATURES], crop["label"]
# Same split as the original notebook (test_size=0.2, random_state=2).
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=2)

baseline = LogisticRegression(random_state=2).fit(Xtr, ytr)  # exactly as shipped in v1
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
cv = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
)
crop_metrics = {
    "rows": int(len(crop)),
    "classes": int(y.nunique()),
    "test_size": int(len(Xte)),
    "v1_logreg_accuracy": round(accuracy_score(yte, baseline.predict(Xte)), 4),
    "rf_accuracy": round(accuracy_score(yte, rf.predict(Xte)), 4),
    "rf_macro_f1": round(f1_score(yte, rf.predict(Xte), average="macro"), 4),
    "rf_cv5_mean": round(float(cv.mean()), 4),
    "rf_cv5_std": round(float(cv.std()), 4),
    "importance": dict(zip(CROP_FEATURES, np.round(rf.feature_importances_, 4).tolist())),
}

# Ship a model fitted on all rows; the scores above come from the held-out split.
final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
classes = final.classes_.tolist()
trees = [export_tree(e, len(classes)) for e in final.estimators_]

ref = final.predict_proba(Xte.values)
mine = np.array([predict_exported(trees, row, len(classes)) for row in Xte.values])
assert np.abs(ref - mine).max() < 1e-3, "exported forest disagrees with sklearn"

profiles = {}
for label, g in crop.groupby("label"):
    profiles[label] = {
        f: {
            "p10": round(float(g[f].quantile(0.10)), 2),
            "p50": round(float(g[f].quantile(0.50)), 2),
            "p90": round(float(g[f].quantile(0.90)), 2),
        }
        for f in CROP_FEATURES
    }
ranges = {f: [round(float(crop[f].min()), 2), round(float(crop[f].max()), 2)] for f in CROP_FEATURES}

(OUT / "crop.json").write_text(json.dumps({
    "features": CROP_FEATURES,
    "classes": classes,
    "trees": trees,
    "profiles": profiles,
    "ranges": ranges,
}, separators=(",", ":")))

# ---------------------------------------------------------- fertilizer model
fert = pd.read_csv(DATA / "fertilizer.csv")
fert.columns = [c.strip() for c in fert.columns]
fert = fert.rename(columns={"Temparature": "Temperature"})
SOILS = sorted(fert["Soil Type"].unique())
CROPS = sorted(fert["Crop Type"].unique())
NUM = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]


def encode(df):
    out = df[NUM].astype(float).copy()
    for s in SOILS:
        out[f"soil={s}"] = (df["Soil Type"] == s).astype(float)
    for c in CROPS:
        out[f"crop={c}"] = (df["Crop Type"] == c).astype(float)
    return out


FX, fy = encode(fert), fert["Fertilizer Name"]
dt_cv = cross_val_score(
    DecisionTreeClassifier(random_state=42), FX, fy,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
)
dt = DecisionTreeClassifier(random_state=42).fit(FX, fy)
fclasses = dt.classes_.tolist()
ftree = export_tree(dt, len(fclasses))
mine = [fclasses[predict_exported([ftree], r, len(fclasses)).argmax()] for r in FX.values]
assert (np.array(mine) == dt.predict(FX.values)).all(), "exported tree disagrees with sklearn"

fert_ranges = {c: [float(fert[c].min()), float(fert[c].max())] for c in NUM}
(OUT / "fertilizer.json").write_text(json.dumps({
    "numeric": NUM,
    "soils": SOILS,
    "crops": CROPS,
    "features": FX.columns.tolist(),
    "classes": fclasses,
    "tree": ftree,
    "ranges": fert_ranges,
}, separators=(",", ":")))

fert_metrics = {
    "rows": int(len(fert)),
    "classes": len(fclasses),
    "dt_cv5_mean": round(float(dt_cv.mean()), 4),
    "dt_cv5_std": round(float(dt_cv.std()), 4),
}

metrics = {"crop": crop_metrics, "fertilizer": fert_metrics}
(ROOT / "ml" / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "metrics.json").write_text(json.dumps(metrics))
print(json.dumps(metrics, indent=2))
print("crop.json", round((OUT / "crop.json").stat().st_size / 1024), "KB")
