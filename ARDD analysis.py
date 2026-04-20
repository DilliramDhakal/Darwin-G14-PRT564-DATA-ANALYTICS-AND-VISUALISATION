"""
Australian Road Death Analysis
"""
import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, chi2_contingency
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

OUT = "plots"
os.makedirs(OUT, exist_ok=True)

C = {"bl":"#378ADD","te":"#1D9E75","co":"#D85A30","pu":"#7F77DD",
"am":"#BA7517","gr":"#888780","re":"#A32D2D"}

def sav(n):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{n}", dpi=130, bbox_inches="tight")
    plt.close()
    print(f" [saved] {OUT}/{n}")

def hr(t):
    print(f"\n{'='*70}\n {t}\n{'='*70}")

def ols_pv(X, y):
    Xm = np.column_stack([np.ones(len(y)), X.values.astype(float)])
    n, p = Xm.shape
    b = np.linalg.lstsq(Xm, y.astype(float), rcond=None)[0]
    r = y.astype(float) - Xm @ b
    s2 = (r @ r) / max(n-p, 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(Xm.T @ Xm)) * s2, 0))
    pv = 2*(1 - stats.t.cdf(np.abs(b / np.where(se<1e-12,1e-12,se)), df=max(n-p,1)))
    return pd.Series(pv[1:], index=X.columns)

# ── 1. LOAD & MERGE ───────────────────────────────────────────────────────────
hr("STEP 1 – LOAD & MERGE")
crashes = pd.read_csv("crashes.csv", encoding="utf-8-sig", low_memory=False)
fat = pd.read_csv("fatalities.csv", encoding="utf-8-sig", low_memory=False)
crashes.columns = crashes.columns.str.strip()
fat.columns = fat.columns.str.strip()
crashes.rename(columns={"Bus \nInvolvement": "Bus Involvement"}, inplace=True, errors='ignore')
df = crashes.merge(fat[["Crash ID", "Road User", "Gender", "Age", "Age Group"]],
                   on="Crash ID", how="left")
print(f"crashes: {crashes.shape} | fatalities: {fat.shape} | merged: {df.shape}")

# ── 2. SKEWNESS & CENTRAL TENDENCY ───────────────────────────────────────────
hr("STEP 2 – SKEWNESS & CENTRAL TENDENCY")
print(f" {'Column':<23} {'Skew':>8} {'Mean':>8} {'Median':>8} {'Mode':>8} Recommendation")
print("-" * 85)
skew_info = {}
for col in ["Number Fatalities", "Speed Limit", "Age"]:
    if col not in df.columns: continue
    s = df[col].dropna()
    if len(s) == 0: continue
    sk = float(skew(s))
    rec = "USE MEAN" if abs(sk) < 0.5 else ("USE MEDIAN (right-skewed)" if sk > 0 else "USE MEDIAN (left-skewed)")
    mode_val = float(s.mode().iloc[0]) if not s.mode().empty else np.nan
    print(f" {col:<23} {sk:>8.3f} {s.mean():>8.2f} {s.median():>8.2f} {mode_val:>8.2f} {rec}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cols_to_plot = ["Number Fatalities", "Speed Limit", "Age"]
colors = [C["bl"], C["te"], C["co"]]
for ax, col, col_c in zip(axes, cols_to_plot, colors):
    if col not in df.columns:
        ax.set_visible(False)
        continue
    d = df[col].dropna()
    ax.hist(d, bins=35, color=col_c, edgecolor="white", alpha=0.85)
    ax.axvline(d.mean(), color="red", ls="--", lw=1.8, label=f"Mean {d.mean():.1f}")
    ax.axvline(d.median(), color="green", ls="--", lw=1.8, label=f"Median {d.median():.1f}")
    ax.set_title(f"{col}\nskew={skew(d):+.2f}", fontsize=9)
    ax.legend(fontsize=7)
plt.suptitle("Step 2 – Skewness & Central Tendency", fontsize=11, fontweight="bold")
sav("01_skewness.png")

# ── 3. DATA CLEANING ─────────────────────────────────────────────────────────
hr("STEP 3 – DATA CLEANING")
df = df.drop(columns=["_id"], errors="ignore")
df = df.replace([-9, "-9"], np.nan)
numeric_cols = ["Speed Limit", "Age"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

binary_cols = ["Bus Involvement", "Heavy Rigid Truck Involvement",
               "Articulated Truck Involvement", "Christmas Period", "Easter Period"]
for c in binary_cols:
    if c in df.columns:
        df[c] = df[c].map({"Yes": 1, "No": 0, 1:1, 0:0})

df = df.dropna(subset=["Number Fatalities"]).reset_index(drop=True)

num_cols = df.select_dtypes(include=np.number).columns
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

df = df.fillna({
    "Road User": "Unknown",
    "Gender": "Unknown",
    "Age Group": "Unknown",
    "National Remoteness Areas": "Unknown",
    "National Road Type": "Unknown",
    "Crash Type": "Single"
})

pre = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {pre - len(df)} | Final shape: {df.shape}")



# ── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
hr("STEP 4 – FEATURE ENGINEERING")
if "Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour
else:
    df["Hour"] = 12
df["Hour"] = df["Hour"].fillna(12)
df["Is_Night"] = df["Hour"].isin(range(0, 6)).astype(int)

if "Dayweek" in df.columns:
    df["Is_Weekend"] = df["Dayweek"].astype(str).str.contains("Sat|Sun|weekend", case=False, na=False).astype(int)
elif "Day of week" in df.columns:
    df["Is_Weekend"] = df["Day of week"].astype(str).str.contains("Sat|Sun|weekend", case=False, na=False).astype(int)
else:
    df["Is_Weekend"] = 0

df["Is_Holiday"] = df[["Christmas Period", "Easter Period"]].max(axis=1).astype(int)
df["Is_HighSpeed"] = (df["Speed Limit"] >= 100).astype(int)
df["Is_VRU"] = df["Road User"].isin(["Pedestrian", "Pedal cyclist", "Motorcycle rider"]).astype(int)
df["Is_Single_Crash"] = (df["Crash Type"].astype(str).str.lower().str.contains("single")).astype(int)
df["Age_Speed"] = df["Age"] * df["Speed Limit"]
df["Night_Speed"] = df["Is_Night"] * df["Speed Limit"]
