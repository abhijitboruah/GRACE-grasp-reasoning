"""
Ablation study on the rule engine: the contribution of each descriptor group.
Starting from the full engine, one information source is removed at a time and
5-fold cross-validation (random_state=42) is re-run on the grasp dataset.

Variants:
  A0  Full engine (shape + aperture + flexion levels)   [reference]
  A1  No shape       (no cylinder/sphere branching)
  A2  No aperture    (grip aperture unavailable)
  A3  No flexion     (flexion levels collapsed to Medium)
  A4  Flexion only   (no shape, no aperture)

Reported metrics: accuracy, macro-F1, balanced accuracy (mean over folds).
"""
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

RANDOM_STATE=42; N_SPLITS=5
df = pd.read_csv("grace_grasp_dataset.csv")
COL = dict(INDEX_PIP="Index_PIP(f/e)",INDEX_DIP="Index_DIP(f/e)",
           MIDDLE_PIP="Middle_PIP(f/e)",MIDDLE_DIP="Middle_DIP(f/e)",
           RING_PIP="Ring_PIP(f/e)",RING_DIP="Ring_DIP(f/e)",
           LITTLE_PIP=" Little_PIP(f/e)",LITTLE_DIP=" Little_DIP(f/e)",
           THUMB_IP="Thumb_IP(f/e)")
df["F_Index"]=(df[COL["INDEX_PIP"]]+df[COL["INDEX_DIP"]])/2
df["F_Middle"]=(df[COL["MIDDLE_PIP"]]+df[COL["MIDDLE_DIP"]])/2
df["F_Ring"]=(df[COL["RING_PIP"]]+df[COL["RING_DIP"]])/2
df["F_Little"]=(df[COL["LITTLE_PIP"]]+df[COL["LITTLE_DIP"]])/2
df["F_Thumb"]=df[COL["THUMB_IP"]]
SENSOR=["F_Thumb","F_Index","F_Middle","F_Ring","F_Little"]
ALLOWED={"Large_Diameter","Thumb_Adducted","Quadpod","Tripod","Medium_Wrap",
         "Small_Diameter","Power_Sphere","Sphere_3_Finger","Sphere_4_Finger"}

def thr_of(tr):
    return {c:(np.quantile(tr[c].astype(float),0.33),
               np.quantile(tr[c].astype(float),0.66)) for c in SENSOR}
def flex(x,lo,hi): return "Low" if x<lo else ("Medium" if x<hi else "High")

def engine(row,thr,use_shape,use_aperture,use_flexion):
    if use_flexion:
        lv={c:flex(float(row[c]),*thr[c]) for c in SENSOR}
    else:
        lv={c:"Medium" for c in SENSOR}     # flexion ablated -> uninformative
    fI,fM,fR,fL="F_Index","F_Middle","F_Ring","F_Little"
    grip=str(row.get("Grip_Aperature","")).strip() if use_aperture else ""
    shape=str(row.get("Shape","")).strip().lower() if use_shape else ""
    active=sum(1 for f in [fI,fM,fR,fL] if lv[f]!="Low")
    thumb=lv["F_Thumb"]!="Low"
    highwrap=all(lv[f]=="High" for f in [fI,fM,fR,fL])
    mediumwrap=sum(1 for f in [fI,fM,fR,fL] if lv[f] in("Medium","High"))>=3
    tripod=(lv[fI] in("Medium","High") and lv[fM] in("Medium","High")
            and lv[fR]=="Low" and lv[fL]=="Low" and thumb)
    pred=None
    if shape=="cylinder":
        if grip=="Maximal" and highwrap: pred="Large_Diameter"
        elif grip=="Intermediate" and highwrap: pred="Small_Diameter"
        elif grip=="Intermediate" and mediumwrap: pred="Medium_Wrap"
        else: pred="Large_Diameter" if grip=="Maximal" else "Medium_Wrap"
    elif shape=="sphere":
        if grip=="Maximal" and active>=4 and thumb and highwrap: pred="Power_Sphere"
        elif tripod and grip in("Minimal","Intermediate"): pred="Tripod"
        elif thumb and active==2 and grip in("Minimal","Intermediate"): pred="Sphere_3_Finger"
        elif thumb and active==3 and grip in("Intermediate","Maximal"): pred="Sphere_4_Finger"
        elif thumb and active==3 and grip in("Minimal","Intermediate"): pred="Quadpod"
        else: pred="Power_Sphere" if grip=="Maximal" else ("Sphere_4_Finger" if active>=3 else "Sphere_3_Finger")
    else:  # no shape info
        if tripod and grip in("Minimal","Intermediate"): pred="Tripod"
        elif grip=="Minimal" and thumb and active<=2: pred="Thumb_Adducted"
        elif grip=="Maximal" and highwrap: pred="Large_Diameter"
        elif grip=="Intermediate" and mediumwrap: pred="Medium_Wrap"
        else: pred="Medium_Wrap" if grip=="Intermediate" else "Small_Diameter"
    return pred if pred in ALLOWED else "Medium_Wrap"

def run(use_shape,use_aperture,use_flexion):
    X=df[SENSOR+["Grip_Aperature","Shape"]]; y=df["Grasp_Type"].astype(str)
    skf=StratifiedKFold(N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    A,F,B=[],[],[]
    for tr,te in skf.split(X,y):
        thr=thr_of(df.iloc[tr])
        preds=[engine(df.iloc[i],thr,use_shape,use_aperture,use_flexion) for i in te]
        yt=y.iloc[te].values
        A.append(accuracy_score(yt,preds)); F.append(f1_score(yt,preds,average="macro"))
        B.append(balanced_accuracy_score(yt,preds))
    return np.mean(A),np.mean(F),np.mean(B)

variants=[
 ("A0  Full (shape+aperture+flexion)",True,True,True),
 ("A1  - shape",                       False,True,True),
 ("A2  - aperture",                    True,False,True),
 ("A3  - flexion levels",              True,True,False),
 ("A4  flexion only (no shape/aper.)", False,False,True),
]
print(f"{'Variant':38s} {'Acc':>7} {'MacroF1':>8} {'BalAcc':>7}")
print("-"*64)
rows=[]
for name,s,a,fl in variants:
    acc,mf,ba=run(s,a,fl)
    rows.append((name,acc,mf,ba))
    print(f"{name:38s} {acc:7.4f} {mf:8.4f} {ba:7.4f}")
import json
json.dump(rows, open("ablation_results.json","w"), indent=2)
