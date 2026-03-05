import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 1. Data load
# ---------------------------------------------------------
file_path = "your_file.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. Data prepare
# ---------------------------------------------------------
# (1) Cluster label matching (0/1 -> Group Name)
cluster_map = {
    0: 'Group A (Mild/Stable)',
    1: 'Group B (Severe/Complex)',
    '0': 'Group A (Mild/Stable)',
    '1': 'Group B (Severe/Complex)'
}

if 'Preop_clustering_2' in df.columns:
    df['Preop_clustering_2'] = df['Preop_clustering_2'].map(lambda x: cluster_map.get(x, x))
else:
    raise ValueError("Preop_clustering_2 does not found.")

# (2) NRS
if 'pain_intensity' in df.columns:
    df['pain_intensity'] = pd.to_numeric(df['pain_intensity'], errors='coerce').fillna(0)
elif 'Intensity_Raw' in df.columns:
    df['pain_intensity'] = pd.to_numeric(df['Intensity_Raw'], errors='coerce').fillna(0)
else:
    raise ValueError("pain_intensity does not found.")

# (3) Pattern missing handling
if 'Pattern' in df.columns:
    df['Pattern'] = df['Pattern'].fillna('Unknown')
else:
    raise ValueError("Pattern does not found.")

# (4) Time_Bin 
if 'Time_Bin' not in df.columns:
    raise ValueError("Time_Bin does not found.")


# Add or remove items from the list as necessary.
bin_order = ['D-3', 'D-2', 'D-1', 'POD #0', 'POD #1', 'POD #2', 'POD #3', 'POD #4', 'POD #5', 'POD #6', 'POD #7']
target_groups = ['Group A (Mild/Stable)', 'Group B (Severe/Complex)']

df_analysis = df[
    (df['Time_Bin'].isin(bin_order)) &
    (df['Preop_clustering_2'].isin(target_groups))
].copy()

print(f"{len(df_analysis)}")

# ---------------------------------------------------------
# 3. Aggregation
# ---------------------------------------------------------
# (1) Bubble chart: Count & Mean Intensity
bubble_agg = df_analysis.groupby(['Preop_clustering_2', 'Time_Bin', 'Pattern']).agg(
    Count=('subject_id', 'count'),
    Mean_Intensity=('pain_intensity', 'mean')
).reset_index()

# (2) Total Count
total_agg = df_analysis.groupby(['Preop_clustering_2', 'Time_Bin'])['subject_id'].count().reset_index(name='Total')

# Percentage
viz_data = bubble_agg.merge(total_agg, on=['Preop_clustering_2', 'Time_Bin'])
viz_data['Percentage'] = (viz_data['Count'] / viz_data['Total']) * 100

# (3) Line chart: 전체 Mean Intensity
line_data = df_analysis.groupby(['Preop_clustering_2', 'Time_Bin'])['pain_intensity'].mean().reset_index()

# ---------------------------------------------------------
# 4. Metric
# ---------------------------------------------------------
# X axis
time_map = {b: i for i, b in enumerate(bin_order)}
viz_data['X_Idx'] = viz_data['Time_Bin'].map(time_map)
line_data['X_Idx'] = line_data['Time_Bin'].map(time_map)

# Y axis
defined_order = [
    'Unspecified', 'Unknown', 'Other', 'Dull', 'Numbing',  
    'Aching', 'Pressure', 'Throbbing',                     
    'Squeezing', 'Burning', 'Sharp', 'Stabbing'           
]


existing_patterns = df_analysis['Pattern'].unique().tolist()
final_pattern_order = [p for p in defined_order if p in existing_patterns]
for p in existing_patterns:
    if p not in final_pattern_order:
        final_pattern_order.append(p)

pattern_map_idx = {p: i for i, p in enumerate(final_pattern_order)}

viz_data['Y_Idx'] = viz_data['Pattern'].map(pattern_map_idx)
viz_data = viz_data.dropna(subset=['Y_Idx'])

# ---------------------------------------------------------
# 5. Dual Axis Bubble Chart
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)

for i, grp in enumerate(target_groups):
    ax1 = axes[i]   
    ax2 = ax1.twinx() 

    
    subset_bubble = viz_data[viz_data['Preop_clustering_2'] == grp]
    subset_line = line_data[line_data['Preop_clustering_2'] == grp].sort_values('X_Idx')

    if subset_bubble.empty: continue

    # --- [1] Bubble Chart (Patterns) ---
    sizes = subset_bubble['Percentage'] * 25

    scatter = ax1.scatter(
        subset_bubble['X_Idx'],
        subset_bubble['Y_Idx'],
        s=sizes,
        c=subset_bubble['Mean_Intensity'],
        cmap='RdYlBu_r',
        vmin=0, vmax=10,
        alpha=0.8,
        edgecolor='gray',
        linewidth=0.5,
        zorder=2
    )

    # Dominant Pattern
    idx_max = subset_bubble.groupby('Time_Bin')['Percentage'].idxmax()
    dominant = subset_bubble.loc[idx_max]

    ax1.scatter(
        dominant['X_Idx'],
        dominant['Y_Idx'],
        s=dominant['Percentage'] * 25,
        facecolors='none',
        edgecolors='black',
        linewidth=2,
        zorder=3,
        label='Dominant Pattern'
    )

    # --- [2] Line Chart (Overall Intensity) ---
    ax2.plot(
        subset_line['X_Idx'],
        subset_line['pain_intensity'],
        color='black',
        linestyle='--',
        marker='x',
        markersize=8,
        linewidth=2,
        label='Overall Mean Intensity',
        alpha=0.6,
        zorder=1
    )

   
    title_bg = '#3498db' if 'Group A' in grp else '#e74c3c'
    ax1.set_title(f'{grp}: Pattern Evolution & Intensity',
                  fontsize=16, fontweight='bold', color='white', backgroundcolor=title_bg, pad=15)

    # Y axis 1
    ax1.set_yticks(range(len(final_pattern_order)))
    ax1.set_yticklabels(final_pattern_order, fontsize=11)
    ax1.set_ylabel('Pain Pattern Types', fontsize=12, fontweight='bold')
    # ax1.grid(True, axis='y', linestyle=':', alpha=0.4)
    ax1.set_ylim(-0.5, len(final_pattern_order) - 0.5)

    # Y axis 2
    ax2.set_ylabel('Overall Mean Intensity (NRS)', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylim(0, 10.5)

    # oper (D-1과 POD#0 사이 = 2.5)
    ax1.axvline(2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.text(2.5, len(final_pattern_order)-0.5, ' Surgery', ha='left', va='top', fontsize=10, fontweight='bold')

    # color bar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.08, aspect=30)
    cbar.set_label('Mean NRS of Each Pattern', rotation=270, labelpad=15)

# X axis
axes[1].set_xticks(range(len(bin_order)))
axes[1].set_xticklabels(bin_order, fontsize=12, fontweight='bold')
axes[1].set_xlabel('Timeline', fontsize=14)

# legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Freq: 20%', markerfacecolor='gray', markersize=math.sqrt(20*25)),
    Line2D([0], [0], marker='o', color='w', label='Freq: 50%', markerfacecolor='gray', markersize=math.sqrt(50*25)),
    Line2D([0], [0], marker='o', color='none', markeredgecolor='black', markeredgewidth=2, label='Dominant Pattern'),
    Line2D([0], [0], color='black', linestyle='--', marker='x', label='Overall Intensity Trend')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.475, 1.03), ncol=4, fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, right=0.85)
plt.savefig('viz_bubble_pattern_evolution_final.png', dpi=300, bbox_inches='tight')
print("viz_bubble_pattern_evolution_final.png")
plt.show()
