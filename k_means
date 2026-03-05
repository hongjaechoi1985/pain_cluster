import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
file_path = "your_file.csv"
df = pd.read_csv(file_path)

print(f"[INFO] Total rows loaded: {len(df)}")

# ---------------------------------------------------------
# 2. Filter Pre-op Data & Handle Missing Values
# ---------------------------------------------------------
# Filter only Pre-op data
preop_df = df[df['hours_from_surgery'] < 0].copy()

# Ensure numeric conversion for Intensity
# We use Intensity_Raw to create the numeric 'pain_intensity' column needed for calculation
if 'Intensity_Raw' in preop_df.columns:
    preop_df['pain_intensity'] = pd.to_numeric(preop_df['Intensity_Raw'], errors='coerce').fillna(0)
else:
    # Fallback if 'pain_intensity' is already numeric in the file
    if 'pain_intensity' in preop_df.columns:
        preop_df['pain_intensity'] = preop_df['pain_intensity'].fillna(0)
    else:
        raise ValueError("Column 'Intensity_Raw' or 'pain_intensity' not found.")

# Fill NaNs for categorical columns to prevent errors during encoding
for col in ['Pattern', 'Frequency', 'Location_std']:
    if col in preop_df.columns:
        preop_df[col] = preop_df[col].fillna('Unknown')
    else:
        raise ValueError(f"Column '{col}' not found in CSV.")

print(f"[INFO] Pre-op rows filtered: {len(preop_df)}")

# ---------------------------------------------------------
# 3. Aggregation by Subject (using Preop_clustering_2)
# ---------------------------------------------------------
# Helper for mode
def mode_or_unknown(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else 'Unknown'

# Check if target cluster column exists
if 'Preop_clustering_2' not in preop_df.columns:
    raise ValueError("Column 'Preop_clustering_2' not found. Please run the clustering code first.")

# Aggregate features + Cluster Label
agg_funcs = {
    'pain_intensity': 'mean',
    'Pattern': mode_or_unknown,
    'Location_std': mode_or_unknown,
    'Frequency': mode_or_unknown,
    'Preop_clustering_2': 'first'  # Take the existing label assigned in the previous step
}

subj_df = preop_df.groupby('subject_id').agg(agg_funcs).reset_index()

# Map numeric labels to readable names for the plot
# 0 -> Group A (Mild/Stable), 1 -> Group B (Severe/Complex)
# (Adjust this mapping if your cluster IDs are swapped)
cluster_label_map = {
    0: 'Group A (Mild/Stable)',
    1: 'Group B (Severe/Complex)',
    -1: 'Unknown'
}
subj_df['Cluster_Label'] = subj_df['Preop_clustering_2'].map(cluster_label_map)

# Filter out Unknowns if any
subj_df = subj_df[subj_df['Preop_clustering_2'] != -1].copy()

print(f"[INFO] Subjects for t-SNE: {len(subj_df)}")

# ---------------------------------------------------------
# 4. t-SNE Execution
# ---------------------------------------------------------
numeric_features = ['pain_intensity']
categorical_features = ['Pattern', 'Location_std', 'Frequency']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Transform data
X = preprocessor.fit_transform(subj_df)
if hasattr(X, "toarray"):
    X = X.toarray()

# Run t-SNE
n_samples = X.shape[0]
perp_val = min(30, n_samples - 1) if n_samples > 1 else 1

tsne = TSNE(n_components=2, perplexity=perp_val, random_state=42, n_iter=1000, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X)

# ---------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------
# Add jitter for better visibility
def add_jitter(arr, amount=0.5):
    np.random.seed(42)
    return arr + np.random.uniform(-amount, amount, len(arr))

subj_df['tsne_1'] = add_jitter(X_embedded[:, 0])
subj_df['tsne_2'] = add_jitter(X_embedded[:, 1])

# copy original
x = subj_df['tsne_1'].values.copy()
y = subj_df['tsne_2'].values.copy()


plt.figure(figsize=(10, 8))

# Define colors
palette = {
    'Group A (Mild/Stable)': '#3498db', # Blue
    'Group B (Severe/Complex)': '#e74c3c' # Red
}

sns.scatterplot(
    x='tsne_1', y='tsne_2',
    hue='Cluster_Label',
    hue_order=['Group A (Mild/Stable)', 'Group B (Severe/Complex)'],
    palette=palette,
    data=subj_df,
    s=100, alpha=0.8, edgecolor='white', linewidth=0.5
)

plt.title('t-SNE Visualization of Pre-op Clusters (Preop_clustering_2)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(False)

plt.tight_layout()
plt.savefig('tsne_preop_clusters_v2.png', dpi=300)
print("Graph saved: tsne_preop_clusters_v2.png")
plt.show()
