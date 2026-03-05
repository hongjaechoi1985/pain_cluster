import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances

# ---------------------------------------------------------
# 1. Data load
# ---------------------------------------------------------
file_path = "your_file.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------
def extract_val(text):
    if pd.isna(text): return "Unknown"
    parts = str(text).split('|')
    return parts[-1].strip() if len(parts) >= 3 else str(text).strip()

# (1) Intensity, Pattern, Frequency
df['Intensity_Raw'] = df['attr2'].apply(extract_val)
df['Pattern_Raw'] = df['attr3'].apply(extract_val)
df['Frequency_Raw'] = df['attr4'].apply(extract_val)

# Pattern
pattern_map = {
    '뻐근한': 'Aching', '욱신욱신 쑤시는': 'Throbbing', '기타': 'Other', '표현 못함': 'Unspecified',
    '저리는 듯한': 'Numbing', '둔한': 'Dull', '날카로운': 'Sharp', '타는 듯한': 'Burning',
    '칼로 베인 듯한': 'Stabbing', '쥐어 짜는 듯한': 'Squeezing', '압박감': 'Pressure'
}
df['Pattern'] = df['Pattern_Raw'].map(lambda x: pattern_map.get(str(x).strip(), x))

# Frequency
frequency_map = {
    '지속적': 'Continuous', '02': 'Continuous', '2': 'Continuous',
    '간헐적': 'Intermittent', '01': 'Intermittent', '1': 'Intermittent',
    '발작적으로 갑자기': 'Intermittent', '기타': 'Other', '표현못함': 'Unspecified'
}
df['Frequency'] = df['Frequency_Raw'].map(lambda x: frequency_map.get(str(x).strip(), x))

# Intensity
df['pain_intensity'] = pd.to_numeric(df['Intensity_Raw'], errors='coerce').fillna(0)

# (2) Location 
if 'Location_std' not in df.columns:
    df['Location_std'] = df['attr1'].apply(extract_val)

# ---------------------------------------------------------
# 3. Pre-op 
# ---------------------------------------------------------

preop_df = df[df['hours_from_surgery'] < 0].copy()

# Aggregation
# Location 
agg_funcs = {
    'pain_intensity': 'mean',
    'Pattern': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'Location_std': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown', 
    'Frequency': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
}

subj_df = preop_df.groupby('subject_id').agg(agg_funcs).reset_index()


# ---------------------------------------------------------
# 4. Vectorization
# ---------------------------------------------------------
numeric_features = ['pain_intensity']

categorical_features = ['Pattern', 'Location_std', 'Frequency']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


X = preprocessor.fit_transform(subj_df)


if hasattr(X, "toarray"):
    X = X.toarray()



# ---------------------------------------------------------
# 5. Metrics (K=2 ~ 7)
# ---------------------------------------------------------
k_range = range(2, 8)

metrics = {
    'Silhouette': [],
    'Calinski-Harabasz': [],
    'Davies-Bouldin': []
}

print("\nCalculating metrics for Pre-op data...")
for k in k_range:
    # K-Means 
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = kmeans.fit_predict(X)

    # Calculation
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    metrics['Silhouette'].append(sil)
    metrics['Calinski-Harabasz'].append(ch)
    metrics['Davies-Bouldin'].append(db)

    print(f"K={k} -> Silhouette: {sil:.4f}, CH: {ch:.1f}, DB: {db:.4f}")

# ---------------------------------------------------------
# 6. STD, visualization
# ---------------------------------------------------------
# Min-Max Scaling
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_metrics = {
    'Silhouette (Higher is better)': normalize(np.array(metrics['Silhouette'])),
    'Calinski-Harabasz (Higher is better)': normalize(np.array(metrics['Calinski-Harabasz'])),
    'Davies-Bouldin (Lower is better)': normalize(np.array(metrics['Davies-Bouldin']))
}


plt.figure(figsize=(10, 8))


markers = ['s', '^', 'D'] 
colors = ['blue', 'green', 'red']
linestyles = ['-', '-', '--']

for i, (name, values) in enumerate(norm_metrics.items()):
    plt.plot(k_range, values, marker=markers[i], linestyle=linestyles[i],
             color=colors[i], linewidth=2.5, markersize=9, label=name)


plt.title('Normalized Cluster Validation (Pre-op Data with Location_std)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Clusters (K)', fontsize=14)
plt.ylabel('Normalized Score (0 to 1)', fontsize=14)
plt.xticks(k_range)
plt.ylim(-0.05, 1.15)
plt.legend(fontsize=11, loc='best', frameon=True, shadow=True)
# plt.grid(True, linestyle='--', alpha=0.4)
plt.grid(False)

plt.tight_layout()
plt.savefig('normalized_validation_preop_location_std.png', dpi=300)
plt.show()
