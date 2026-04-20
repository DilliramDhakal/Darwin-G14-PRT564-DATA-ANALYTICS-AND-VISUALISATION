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

# ── 5. ENCODING ─────────────────────────────────────────────────────────────
hr("STEP 5 – ENCODING")
cat_cols = ["State", "Road User", "Gender", "National Remoteness Areas", "National Road Type"]
for col in cat_cols:
    if col in df.columns:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))

features = ["Speed Limit", "Age", "Hour", "Is_Night", "Is_Weekend", "Is_HighSpeed",
            "Age_Speed", "Night_Speed", "State_enc"]
features = [f for f in features if f in df.columns]

dataset = df[features + ["Is_VRU", "Is_Single_Crash", "Number Fatalities"]].dropna().copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset[features])

# ── 6. PCA ───────────────────────────────────────────────────────────────────
hr("STEP 6 – PCA")

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA Components: {pca.n_components_}")
print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.4f}")

hr("PCA - Explained Variance (PC1 & PC2)")

explained_variance_ratio = pca.explained_variance_ratio_

print("PCA Explained Variance by Component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var*100:.2f}%")

print("\n=== Focus on PC1 and PC2 ===")
print(f"PC1 Explained Variance : {explained_variance_ratio[0]*100:.2f}%")
print(f"PC2 Explained Variance : {explained_variance_ratio[1]*100:.2f}%")
print(f"Combined PC1 + PC2     : {(explained_variance_ratio[0] + explained_variance_ratio[1])*100:.2f}%")

cumulative_variance = np.cumsum(explained_variance_ratio)
print(f"\nCumulative Variance up to PC2: {cumulative_variance[1]*100:.2f}%")
print(f"Total Variance explained by {pca.n_components_} components: {cumulative_variance[-1]*100:.2f}%")

# ── RQ1 DESCRIPTIVE ─────────────────────────────────────────────────────────
hr("RQ1 – Fatalities by state, road-user group & remoteness")
if "Year" in fat.columns and "State" in fat.columns:
    yr = fat.groupby(["Year", "State"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, st in enumerate(sorted(yr["State"].unique())):
        sub = yr[yr["State"] == st]
        ax.plot(sub["Year"], sub["n"], marker="o", ms=3, lw=1.5,
                label=st, color=list(C.values())[i % len(C)])
    ax.legend(fontsize=8, ncol=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Fatalities")
    ax.set_title("RQ1a – Annual fatalities by state", fontweight="bold")
    sav("05_RQ1a_trend.png")

    ru = fat["Road User"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(ru.index[::-1], ru.values[::-1], color=list(C.values())[:len(ru)], edgecolor="white")
    axes[0].set_title("RQ1b – Fatalities by road user type")
    for i, (v, l) in enumerate(zip(ru.values[::-1], ru.index[::-1])):
        axes[0].text(v + 80, i, f"{v:,} ({v/ru.sum()*100:.1f}%)", va="center", fontsize=8)

    rem_pie = fat["National Remoteness Areas"].replace({"Unknown": np.nan}).dropna().value_counts()
    axes[1].pie(rem_pie.values, labels=rem_pie.index, autopct="%1.1f%%", startangle=140,
                colors=[C["bl"], C["te"], C["am"], C["co"], C["re"]])
    axes[1].set_title("RQ1c – By remoteness")
    plt.suptitle("RQ1 – Road user & remoteness", fontsize=12, fontweight="bold")
    sav("06_RQ1bc.png")

# ── RQ1d – Crashes by Age, Speed Limit & Road Type ───────────────────────────
hr("RQ1d – Number of Crashes by Age, Speed Limit & Road Type")

# Create Age Groups and Speed Bands
df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[0, 18, 25, 35, 45, 55, 65, 100],
                         labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                         right=False)

df['Speed_Band'] = pd.cut(df['Speed Limit'], 
                          bins=[0, 50, 80, 100, 130], 
                          labels=['≤50', '51-80', '81-100', '>100'],
                          right=True)

# Ensure Road Type column exists
road_col = "National Road Type" if "National Road Type" in df.columns else "Road Type"

# 1. Main Heatmap: Age vs Speed Limit (Overall)
crash_heatmap = df.pivot_table(
    values='Crash ID',
    index='Age_Group',
    columns='Speed_Band',
    aggfunc='nunique',
    fill_value=0
)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(crash_heatmap, annot=True, fmt=',', cmap='YlOrRd', linewidths=0.5, 
            cbar_kws={'label': 'Number of Crashes'}, ax=ax)
ax.set_title('RQ1d – Number of Crashes by Age Group and Speed Limit', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Speed Limit Band (km/h)')
ax.set_ylabel('Age Group')
sav("18_RQ1d_Age_Speed_Heatmap.png")

# 2. Road Type Interaction - Average Fatalities Heatmap by Road Type
# Group by Road Type, Age Group, and Speed Band
road_age_speed = df.groupby([road_col, 'Age_Group', 'Speed_Band']).agg(
    Num_Crashes=('Crash ID', 'nunique'),
    Avg_Fatalities=('Number Fatalities', 'mean')
).reset_index()

# Pivot for heatmap per major road type (Top 5 most common road types)
top_road_types = df[road_col].value_counts().head(5).index.tolist()

fig, axes = plt.subplots(1, len(top_road_types), figsize=(18, 8), sharey=True)

for i, road in enumerate(top_road_types):
    subset = road_age_speed[road_age_speed[road_col] == road]
    pivot = subset.pivot_table(
        values='Num_Crashes',
        index='Age_Group',
        columns='Speed_Band',
        aggfunc='sum',
        fill_value=0
    )
    
    sns.heatmap(pivot, annot=True, fmt=',', cmap='YlOrRd', linewidths=0.5, ax=axes[i])
    axes[i].set_title(f'{road}', fontsize=11)
    axes[i].set_xlabel('Speed Band')
    if i == 0:
        axes[i].set_ylabel('Age Group')

plt.suptitle('RQ1d – Number of Crashes by Age, Speed Limit & Road Type\n(Top 5 Road Types)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
sav("19_RQ1d_Age_Speed_RoadType_Heatmaps.png")

print("\nAge + Speed Limit + Road Type analysis completed!")
print("   → 18_RQ1d_Age_Speed_Heatmap.png")
print("   → 19_RQ1d_Age_Speed_RoadType_Heatmaps.png")
# ── RQ2 – VRU CLASSIFICATION ────────────────────────────────────────────────
print("\n=== RQ2: VRU (Vulnerable Road User) Classification ===")
X = X_pca
y = dataset["Is_VRU"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, pred))
print("AUC :", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, pred))