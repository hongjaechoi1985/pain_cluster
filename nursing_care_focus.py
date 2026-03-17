import matplotlib.pyplot as plt
import seaborn as sns

NANDA_DOMAIN_LABEL = {
    1: "1. Pain/Comfort",
    2: "2. Respiration/Airway",
    3: "3. Circulation/Perfusion",
    4: "4. Infection/Protection",
    5: "5. Elimination/Digestion",
    6: "6. Fluids/Metabolism",
    7: "7. Cognition/Emotion/Function"
}

# 1. Merge cluster info
cluster_info = df[['subject_id', 'Preop_clustering']].drop_duplicates()
pivot_with_cluster = pivot_prop.merge(cluster_info, on='subject_id', how='left')

# 2. Set plot style and colors
sns.set_theme(style="white")

# Use distinct colors for the 7 domains
palette = sns.color_palette("Set1", n_colors=len(NANDA_DOMAIN_LABEL))
color_map = dict(zip(sorted(NANDA_DOMAIN_LABEL.keys()), palette))

# 3. Create subplots for each cluster
groups = sorted(pivot_with_cluster['Preop_clustering'].unique())
fig, axes = plt.subplots(1, len(groups), figsize=(18, 7), sharey=True)

day_axis = range(-3, 8)
day_labels = [f"D{d}" if d < 0 else f"POD #{d}" for d in day_axis]
domain_cols = sorted(NANDA_DOMAIN_LABEL.keys()) 

for i, group in enumerate(groups):
    ax = axes[i]
    group_data = pivot_with_cluster[pivot_with_cluster['Preop_clustering'] == group]

    # Calculate daily mean proportions
    plot_data = group_data.groupby('day_diff')[domain_cols].mean().reindex(day_axis).fillna(0)

    for col in domain_cols:
        ax.plot(plot_data.index, plot_data[col],
                label=NANDA_DOMAIN_LABEL[col],
                color=color_map[col],
                marker='o', markersize=5, linewidth=2.5, alpha=0.8)

    # Format plot
    ax.set_xlabel("Days from Surgery", fontsize=13)
    if i == 0:
        ax.set_ylabel("Mean Proportion", fontsize=13)

    # Highlight surgery day (Day 0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.2, ax.get_ylim()[1] * 0.9, 'Surgery', color='gray', fontweight='bold')

    ax.set_xticks(day_axis)
    ax.set_xticklabels(day_labels)
    ax.tick_params(axis='x', which='both', bottom=True)
    ax.tick_params(axis='y', which='both', left=True)
    ax.grid(False)

# Add global legend at the top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=4, fontsize=11, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(
    "nanda_domain_trajectory_by_cluster.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
