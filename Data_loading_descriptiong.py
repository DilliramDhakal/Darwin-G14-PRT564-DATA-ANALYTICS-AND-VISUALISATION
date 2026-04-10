# ==================== 1. DATA LOADING & DESCRIPTION ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the two ARDD files (update paths to your downloaded files)
fatalities = pd.read_csv('fatalities.csv')      # main file with Age, Road User, etc.
crashes = pd.read_csv('crashes.csv')            # crash-level info

# Merge on Crash ID
df = pd.merge(fatalities, crashes, on='Crash ID', how='left')

print("Dataset shape:", df.shape)
print(df.info())
print(df.describe())

# Quick target distribution (Age)
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age (Target Variable)')
plt.show()

