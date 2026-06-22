"""
Grasp interpretation experiment for the GRACE framework.

Evaluates the ontology-driven rule engine against eight machine-learning
classifiers (logistic regression, decision tree, random forest, extra trees,
gradient boosting, XGBoost, LightGBM, and a shallow MLP) on the grasp
dataset, under sensor-only and sensor-plus-context feature configurations.

Outputs:
  - predictive performance (accuracy, macro-F1, balanced accuracy)
  - ontology diagnostics (leaf coverage, constraint-violation rate,
    prediction stability under 2% Gaussian noise)
  - per-constraint violation breakdown (C1, C2, C3)
  - runtime per sample for each model
  - paired t-tests between the rule engine and the reference classifier
  - class distribution

Protocol: 5-fold stratified cross-validation (random_state=42), with
quantile-based flexion discretization (33rd and 66th percentiles) computed
on the training fold only.
"""

import os, time, json, sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Optional gradient-boosting libraries; skipped if not installed.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Configuration
DATA_PATH    = "grace_grasp_dataset.csv"
TARGET_COL   = "Grasp_Type"
N_SPLITS     = 5
RANDOM_STATE = 42
OUTDIR       = "outputs_truth"
os.makedirs(OUTDIR, exist_ok=True)

# Joint-angle columns.
COL = dict(
    INDEX_PIP="Index_PIP(f/e)",   INDEX_DIP="Index_DIP(f/e)",
    MIDDLE_PIP="Middle_PIP(f/e)", MIDDLE_DIP="Middle_DIP(f/e)",
    RING_PIP="Ring_PIP(f/e)",     RING_DIP="Ring_DIP(f/e)",
    LITTLE_PIP=" Little_PIP(f/e)", LITTLE_DIP=" Little_DIP(f/e)",
    THUMB_IP="Thumb_IP(f/e)",
)

# 'Curvature' is excluded as it is constant across the dataset.
CONTEXT_COLS = ["Grip_Aperature", "Shape", "Material", "Tactility", "Object"]
SENSOR_COLS  = ["F_Thumb", "F_Index", "F_Middle", "F_Ring", "F_Little"]

# The nine grasp-type labels.
ALLOWED = {"Large_Diameter", "Thumb_Adducted", "Quadpod", "Tripod",
           "Medium_Wrap", "Small_Diameter", "Power_Sphere",
           "Sphere_3_Finger", "Sphere_4_Finger"}

# Integer label encoding for XGBoost and LightGBM.
_SORTED_LABELS = sorted(ALLOWED)
LABEL2ID = {lab: i for i, lab in enumerate(_SORTED_LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# Helper functions
def add_distal_aggregates(df):
    df = df.copy()
    df["F_Index"]  = (df[COL["INDEX_PIP"]]  + df[COL["INDEX_DIP"]])  / 2.0
    df["F_Middle"] = (df[COL["MIDDLE_PIP"]] + df[COL["MIDDLE_DIP"]]) / 2.0
    df["F_Ring"]   = (df[COL["RING_PIP"]]   + df[COL["RING_DIP"]])   / 2.0
    df["F_Little"] = (df[COL["LITTLE_PIP"]] + df[COL["LITTLE_DIP"]]) / 2.0
    df["F_Thumb"]  = df[COL["THUMB_IP"]]
    return df

def quantile_thr(train_df):
    """33rd / 66th percentile per sensor feature, TRAIN FOLD ONLY."""
    return {c: (np.quantile(train_df[c].astype(float).values, 0.33),
                np.quantile(train_df[c].astype(float).values, 0.66))
            for c in SENSOR_COLS}

def flex(x, low, high):
    return "LowFlexion" if x < low else ("MediumFlexion" if x < high else "HighFlexion")

def rule_predict(df_part, thr):
    """Deterministic rule engine mapping each instance to one of the nine grasp labels."""
    fI, fM, fR, fL = "F_Index", "F_Middle", "F_Ring", "F_Little"
    preds = []
    for _, r in df_part.iterrows():
        lv = {c: flex(float(r[c]), *thr[c]) for c in SENSOR_COLS}
        grip  = str(r.get("Grip_Aperature", "")).strip()
        shape = str(r.get("Shape", "")).strip().lower()
        active = sum(1 for f in [fI, fM, fR, fL] if lv[f] != "LowFlexion")
        thumb  = (lv["F_Thumb"] != "LowFlexion")
        highwrap   = all(lv[f] == "HighFlexion" for f in [fI, fM, fR, fL])
        mediumwrap = sum(1 for f in [fI, fM, fR, fL]
                         if lv[f] in ("MediumFlexion", "HighFlexion")) >= 3
        tripod = (lv[fI] in ("MediumFlexion", "HighFlexion") and
                  lv[fM] in ("MediumFlexion", "HighFlexion") and
                  lv[fR] == "LowFlexion" and lv[fL] == "LowFlexion" and thumb)
        pred = None
        if shape == "cylinder":
            if   grip == "Maximal"      and highwrap:   pred = "Large_Diameter"
            elif grip == "Intermediate" and highwrap:   pred = "Small_Diameter"
            elif grip == "Intermediate" and mediumwrap: pred = "Medium_Wrap"
            else: pred = "Large_Diameter" if grip == "Maximal" else "Medium_Wrap"
        elif shape == "sphere":
            if   grip == "Maximal" and active >= 4 and thumb and highwrap: pred = "Power_Sphere"
            elif tripod and grip in ("Minimal", "Intermediate"):           pred = "Tripod"
            elif thumb and active == 2 and grip in ("Minimal", "Intermediate"): pred = "Sphere_3_Finger"
            elif thumb and active == 3 and grip in ("Intermediate", "Maximal"): pred = "Sphere_4_Finger"
            elif thumb and active == 3 and grip in ("Minimal", "Intermediate"): pred = "Quadpod"
            else:
                pred = ("Power_Sphere" if grip == "Maximal"
                        else ("Sphere_4_Finger" if active >= 3 else "Sphere_3_Finger"))
        else:
            if   tripod and grip in ("Minimal", "Intermediate"):       pred = "Tripod"
            elif grip == "Minimal" and thumb and active <= 2:          pred = "Thumb_Adducted"
            elif grip == "Maximal" and highwrap:                       pred = "Large_Diameter"
            elif grip == "Intermediate" and mediumwrap:                pred = "Medium_Wrap"
            else: pred = "Medium_Wrap" if grip == "Intermediate" else "Small_Diameter"
        preds.append(pred if pred in ALLOWED else "Medium_Wrap")
    return np.array(preds, dtype=object)

def compute_violations(df_test, y_pred):
    counts = Counter(); violated = 0
    precision = {"Tripod", "Quadpod", "Sphere_3_Finger", "Thumb_Adducted"}
    power     = {"Large_Diameter", "Small_Diameter", "Medium_Wrap",
                 "Power_Sphere", "Sphere_4_Finger"}
    for i, pred in enumerate(y_pred):
        shape = str(df_test.iloc[i]["Shape"]).strip()
        grip  = str(df_test.iloc[i]["Grip_Aperature"]).strip()
        v = []
        if pred in precision and grip == "Maximal" and shape == "Cylinder": v.append("C1")
        if pred in power and grip == "Minimal":                             v.append("C2")
        if shape == "Cylinder" and pred in {"Sphere_3_Finger", "Thumb_Adducted"}: v.append("C3")
        if v: violated += 1; counts.update(v)
    return 100.0 * violated / len(y_pred), counts

def inject_noise(X, cols, pct, rng):
    Xn = X.copy()
    for c in cols:
        col = Xn[c].astype(float).values
        rng_span = float(col.max() - col.min())
        Xn[c] = col + rng.normal(0.0, pct * rng_span, size=len(col))
    return Xn

def metrics(yt, yp):
    return {"Accuracy":    accuracy_score(yt, yp),
            "MacroF1":     f1_score(yt, yp, average="macro"),
            "BalancedAcc": balanced_accuracy_score(yt, yp)}

def get_models():
    """
    Seven classifiers spanning distinct inductive biases:
      - Linear            : Logistic Regression
      - Single tree       : Decision Tree
      - Bagged ensembles  : Random Forest, Extra Trees
      - Boosted ensembles : Gradient Boosting, XGBoost, LightGBM
      - Neural            : MLP (shallow ANN)
    XGBoost and LightGBM are included only if installed.
    """
    models = {
        "LR":  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DT":  DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RF":  RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                      n_jobs=-1),
        "ET":  ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                    n_jobs=-1),
        "GB":  GradientBoostingClassifier(random_state=RANDOM_STATE),
        "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                             random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.3,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    if HAS_LGBM:
        models["LGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=63,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return models


# Display names + a fixed print/table order
MODEL_DISPLAY = {
    "RuleEngine": "Rule Engine", "LR": "Logistic Reg.", "DT": "Decision Tree",
    "RF": "Random Forest", "ET": "Extra Trees", "GB": "Gradient Boost",
    "XGB": "XGBoost", "LGBM": "LightGBM", "ANN": "ANN",
}
MODEL_ORDER = ["RuleEngine", "LR", "DT", "RF", "ET", "GB", "XGB", "LGBM", "ANN"]

def time_ms(fn, X, reps=3):
    n = len(X); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(X); ts.append(time.perf_counter() - t0)
    return (np.mean(ts) / n) * 1000.0

# Cross-validation
def run_cv(df, use_context):
    tag   = "sensor+context" if use_context else "sensor-only"
    Xcols = SENSOR_COLS + (CONTEXT_COLS if use_context else [])
    X = df[Xcols].copy()
    y = df[TARGET_COL].astype(str).copy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pre = ColumnTransformer([
        ("num", StandardScaler(), SENSOR_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         CONTEXT_COLS if use_context else []),
    ])
    models = get_models()
    allm = {"RuleEngine": []}
    for m in models: allm[m] = []
    cov, viol, viol_counts, stab = [], [], [], []
    rt = {n: [] for n in list(models) + ["RuleEngine"]}

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()
        thr = quantile_thr(Xtr)

        yp_rule = rule_predict(Xte, thr)
        allm["RuleEngine"].append(metrics(yte, yp_rule))
        rt["RuleEngine"].append(time_ms(lambda Z: rule_predict(Z, thr), Xte))

        cov.append(100.0 * sum(1 for p in yp_rule if p in ALLOWED) / len(yp_rule))

        if use_context:
            vr, vc = compute_violations(Xte, yp_rule)
        else:
            vr, vc = 0.0, Counter()
        viol.append(vr); viol_counts.append(vc)

        rng = np.random.default_rng(1000 + fold)
        Xte_n = inject_noise(Xte, SENSOR_COLS, 0.02, rng)
        if use_context:
            for c in CONTEXT_COLS: Xte_n[c] = Xte[c].values
        stab.append(100.0 * (yp_rule == rule_predict(Xte_n, thr)).mean())

        for name, clf in models.items():
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            if name in ("XGB", "LGBM"):
                # These libraries need integer-encoded targets.
                ytr_enc = ytr.map(LABEL2ID).values
                pipe.fit(Xtr, ytr_enc)
                yp_enc = pipe.predict(Xte)
                yp = pd.Series(yp_enc).map(ID2LABEL).values
                allm[name].append(metrics(yte, yp))
                rt[name].append(time_ms(pipe.predict, Xte))
            else:
                pipe.fit(Xtr, ytr)
                allm[name].append(metrics(yte, pipe.predict(Xte)))
                rt[name].append(time_ms(pipe.predict, Xte))

        best_ml = max((m for m in allm if m != "RuleEngine"),
                      key=lambda m: allm[m][-1]["Accuracy"])
        print(f"  [{tag}] fold {fold}/{N_SPLITS}  "
              f"Rule={allm['RuleEngine'][-1]['Accuracy']:.4f}  "
              f"best_ML({best_ml})={allm[best_ml][-1]['Accuracy']:.4f}")
        sys.stdout.flush()

    def summ(md):
        return {k: (float(np.mean([m[k] for m in md])),
                    float(np.std([m[k] for m in md]))) for k in md[0]}
    pred = {n: summ(ms) for n, ms in allm.items()}

    # paired t-test: Rule Engine vs RF, and vs the best-performing ML model
    def paired(ml_name):
        out = {}
        for k in ["Accuracy", "MacroF1", "BalancedAcc"]:
            a = [m[k] for m in allm[ml_name]]; b = [m[k] for m in allm["RuleEngine"]]
            t, p = stats.ttest_rel(a, b)
            diff = np.array(a) - np.array(b)
            d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("inf")
            out[k] = {"t": float(t), "p": float(p), "d": d,
                      "ml_mean": float(np.mean(a)), "rule_mean": float(np.mean(b))}
        return out

    best_ml = max((m for m in allm if m != "RuleEngine"),
                  key=lambda m: np.mean([x["Accuracy"] for x in allm[m]]))
    tt    = paired("RF") if "RF" in allm else paired(best_ml)
    tt_best = paired(best_ml)

    total_vc = Counter()
    for c in viol_counts: total_vc.update(c)

    return {"tag": tag, "pred": pred,
            "coverage": (float(np.mean(cov)), float(np.std(cov))),
            "violation_rate": (float(np.mean(viol)), float(np.std(viol))),
            "stability": (float(np.mean(stab)), float(np.std(stab))),
            "viol_counts_total": dict(total_vc),
            "runtime": {n: (float(np.mean(v)), float(np.std(v))) for n, v in rt.items()},
            "ttest": tt, "ttest_best": tt_best, "best_ml": best_ml}

# Output formatting
def print_pred_table(res):
    print(f"\n--- Predictive performance [{res['tag']}] ---")
    print(f"{'Model':<16}{'Accuracy':<22}{'Macro-F1':<22}{'Balanced Acc.':<22}")
    for k in MODEL_ORDER:
        if k not in res["pred"]:
            continue
        s = res["pred"][k]
        print(f"{MODEL_DISPLAY[k]:<16}"
              f"{s['Accuracy'][0]:.4f} ± {s['Accuracy'][1]:.4f}     "
              f"{s['MacroF1'][0]:.4f} ± {s['MacroF1'][1]:.4f}     "
              f"{s['BalancedAcc'][0]:.4f} ± {s['BalancedAcc'][1]:.4f}")

def print_diag(res):
    c = res["coverage"]; v = res["violation_rate"]; s = res["stability"]
    print(f"\n--- Ontology diagnostics [{res['tag']}] ---")
    print(f"Leaf coverage    : {c[0]:.2f} ± {c[1]:.2f} %")
    print(f"Violation rate   : {v[0]:.2f} ± {v[1]:.2f} %")
    print(f"Stability @2%    : {s[0]:.2f} ± {s[1]:.2f} %")
    print(f"Violation counts : {res['viol_counts_total']}")

def print_runtime(res):
    print(f"\n--- Runtime ms/sample [{res['tag']}] ---")
    for n in MODEL_ORDER:
        if n not in res["runtime"]:
            continue
        m, sd = res["runtime"][n]
        print(f"  {MODEL_DISPLAY[n]:<16} {m:.4f} ± {sd:.4f}")

def print_ttest(res):
    print(f"\n--- Paired t-test: Rule Engine vs RF [{res['tag']}] ---")
    for k, r in res["ttest"].items():
        sig = "*" if r["p"] < 0.05 else ""
        print(f"  {k:<13} t={r['t']:.3f}  p={r['p']:.3e}{sig}  d={r['d']:.2f}  "
              f"ML_mean={r['ml_mean']:.4f}")
    bm = res.get("best_ml")
    if bm and bm != "RF":
        print(f"  (best ML overall = {MODEL_DISPLAY.get(bm, bm)})")
        for k, r in res["ttest_best"].items():
            sig = "*" if r["p"] < 0.05 else ""
            print(f"  vs {bm:<7} {k:<11} t={r['t']:.3f}  p={r['p']:.3e}{sig}  "
                  f"d={r['d']:.2f}  ML_mean={r['ml_mean']:.4f}")

# Main
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = add_distal_aggregates(df)
    df = df[SENSOR_COLS + CONTEXT_COLS + [TARGET_COL]].dropna().reset_index(drop=True)

    print("=" * 60)
    print(f"Dataset rows : {len(df)}")
    print(f"Classes ({df[TARGET_COL].nunique()}): "
          f"{dict(Counter(df[TARGET_COL]).most_common())}")
    print(f"XGBoost available : {HAS_XGB}   |   LightGBM available : {HAS_LGBM}")
    if not HAS_XGB:
        print("  (XGBoost not installed; skipped. Install with: pip install xgboost)")
    if not HAS_LGBM:
        print("  (LightGBM not installed; skipped. Install with: pip install lightgbm)")
    print("=" * 60)

    print("\n### Running SENSOR-ONLY ###")
    r_sensor  = run_cv(df, use_context=False)
    print("\n### Running SENSOR + CONTEXT ###")
    r_context = run_cv(df, use_context=True)

    for res in (r_sensor, r_context):
        print_pred_table(res); print_diag(res); print_runtime(res); print_ttest(res)

    with open(os.path.join(OUTDIR, "outputs.json"), "w") as f:
        json.dump({"n_rows": len(df),
                   "class_counts": dict(Counter(df[TARGET_COL])),
                   "sensor_only": r_sensor,
                   "sensor_context": r_context}, f, indent=2)
    print(f"\nSaved -> {OUTDIR}/outputs.json   (send me this file)")
