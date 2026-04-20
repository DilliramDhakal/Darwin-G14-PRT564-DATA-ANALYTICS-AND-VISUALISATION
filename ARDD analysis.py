#### DATA LOADING & MERGE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

fatalities = pd.read_csv('fatalities.csv', low_memory=False)
crashes = pd.read_csv('crashes.csv', low_memory=False)

df = pd.merge(fatalities, crashes, on='Crash ID', how='left')
print("Merged shape:", df.shape)

############### CLEANING #############
df.replace(-9, np.nan, inplace=True)

# === CRITICAL FIX: Clean Speed Limit after merge ===
print("Cleaning Speed Limit column...")
df['Speed Limit_x'] = df['Speed Limit_x'].replace('<40', 35)          # reasonable value
df['Speed Limit_x'] = df['Speed Limit_x'].replace('Unspecified', np.nan)
df['Speed Limit_x'] = pd.to_numeric(df['Speed Limit_x'], errors='coerce')

df['Speed Limit_y'] = df['Speed Limit_y'].replace('<40', 35)
df['Speed Limit_y'] = df['Speed Limit_y'].replace('Unspecified', np.nan)
df['Speed Limit_y'] = pd.to_numeric(df['Speed Limit_y'], errors='coerce')

####check skewness ##########
# Select only numeric columns
numeric_data = df.select_dtypes(include=['number'])

# Compute skewness for each numeric column
skewness = numeric_data.skew()

print("=== SKEWNESS ANALYSIS FOR NUMERIC COLUMNS ===\n")

# Loop through each numeric column and create histogram + skewness value
for col in numeric_data.columns:
    skew_val = skewness[col]
    
    print(f"{col:25} → Skewness: {skew_val:6.3f} ", end="")
    
    # if skew_val > 0.5:
    #     print("→ Strongly Right-Skewed → Recommend MEDIAN imputation")
    # elif skew_val < -0.5:
    #     print("→ Strongly Left-Skewed → Recommend MEDIAN imputation")
    # else:
    #     print("→ Roughly Symmetric → Mean is acceptable")
    
    # Plot histogram
    plt.figure(figsize=(8, 5))
    numeric_data[col].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}\nSkewness = {skew_val:.3f}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    
    # Add vertical lines for Mean and Median
    mean_val = numeric_data[col].mean()
    median_val = numeric_data[col].median()
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='solid', linewidth=2, label=f'Median = {median_val:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Median imputation
num_cols = ['Age', 'Speed Limit_x', 'Speed Limit_y', 'Number Fatalities']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Mode imputation for categorical
cat_cols = ['State_x', 'State_y', 'Crash Type_x', 'Crash Type_y', 'Road User',
            'National Remoteness Areas_x', 'National Remoteness Areas_y',
            'Dayweek', 'Time of day', 'Christmas Period', 'Easter Period']

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"Shape after cleaning: {df.shape}")
print(f"Missing values left: {df.isnull().sum().sum()}\n")

############ 3. FEATURE ENGINEERING + STANDARDISATION ##############
cat_features = [col for col in ['State_x', 'State_y', 'Crash Type_x', 'Crash Type_y',
                                'Road User', 'National Remoteness Areas_x',
                                'National Remoteness Areas_y', 'Dayweek',
                                'Time of day', 'Christmas Period', 'Easter Period']
                if col in df.columns]

X_cat = pd.get_dummies(df[cat_features], drop_first=True)

num_features = ['Speed Limit_x', 'Speed Limit_y', 'Year', 'Month']
X_num = df[[col for col in num_features if col in df.columns]]

X = pd.concat([X_cat, X_num], axis=1)

y_age = df['Age']

# Standardisation
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

print(f"Final feature matrix shape: {X_std.shape}")
print("Standardisation completed successfully!")

# FEATURE SELECTION (Backward Elimination)
print("Feature Selection using Backward Elimination...")

X_const = sm.add_constant(X_std)

def backward_elimination(X, y, threshold=0.05):
    cols = list(X.columns)
    while len(cols) > 0:
        model = sm.OLS(y, X[cols]).fit()
        pvals = model.pvalues[1:]
        max_p = pvals.max()
        if max_p > threshold:
            remove = pvals.idxmax()
            cols.remove(remove)
        else:
            break
    return cols

selected_features = backward_elimination(X_const, y_age)
print("Selected features:", selected_features, "\n")

#############  PCA VISUALISATION

print("PCA Visualisation (PC1 vs PC2)...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_age, cmap='viridis', alpha=0.7)
plt.colorbar(label='Age of Deceased')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: PC1 vs PC2 coloured by Age')
plt.savefig('pca_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"PC1 + PC2 explained variance: {pca.explained_variance_ratio_.sum():.1%}\n")