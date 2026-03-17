import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import re

# =========================
# Settings
# =========================
TOP_N = 10
file_path = 'your_file.csv'
save_dir  = '/save_dir/'
all_days  = list(range(-3, 8))   # D-3 to POD7

# =========================
# 1. Load Data
# =========================
df = pd.read_csv(file_path)

need_cols = ['Group_Label', 'Day_Num', 'note_text', 'Count']
missing_cols = [c for c in need_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df['Count'] = pd.to_numeric(df['Count'], errors='coerce').fillna(0)
df['Day_Num'] = pd.to_numeric(df['Day_Num'], errors='coerce')

# =========================
# 2. Select Top-N Notes (Based on Total Sum)
# =========================
total_counts = df.groupby('note_text', as_index=False)['Count'].sum()
top_notes = total_counts.sort_values('Count', ascending=False).head(TOP_N)['note_text'].tolist()

print(f"Top {TOP_N} notes:", top_notes)
df_top = df[df['note_text'].isin(top_notes)].copy()

# =========================
# 3. Denominator: Total daily records per group
# =========================
day_total = df.groupby(['Group_Label', 'Day_Num'], as_index=False)['Count'].sum()
day_total.rename(columns={'Count': 'DayTotal_AllNotes'}, inplace=True)

# =========================
# 4. Calculate Count and Daily Percentage (%)
# =========================
gdn = df_top.groupby(['Group_Label', 'Day_Num', 'note_text'], as_index=False)['Count'].sum()
gdn.rename(columns={'Count': 'NoteCount'}, inplace=True)

gdn = gdn.merge(day_total, on=['Group_Label', 'Day_Num'], how='left')
gdn['Pct_Day'] = np.where(gdn['DayTotal_AllNotes'] > 0, (gdn['NoteCount'] / gdn['DayTotal_AllNotes']) * 100, 0)
gdn['Day_Num'] = gdn['Day_Num'].astype(int)

# =========================
# 5. Wide Pivot (Count / Pct_Day) by Group
# =========================
def make_wide(df_long, group_label, value_col):
    sub = df_long[df_long['Group_Label'] == group_label]
    return (sub.pivot_table(index='note_text', columns='Day_Num', values=value_col, aggfunc='sum', fill_value=0)
               .reindex(index=top_notes, columns=all_days, fill_value=0))

groups = sorted(df['Group_Label'].dropna().unique())

for g in groups:
    wide_count = make_wide(gdn, g, 'NoteCount')
    wide_pct   = make_wide(gdn, g, 'Pct_Day')

    print(f"\n=== [{g}] Daily Count (Top {TOP_N}) ===")
    print(wide_count.to_string())

    print(f"\n=== [{g}] Daily Percentage (%) (Top {TOP_N}) ===")
    print(wide_pct.round(2).to_string())

    # Sanitize group names for safe file saving
    safe_g = re.sub(r'[\\/*?:"<>| ()]', '_', g).strip('_')
    wide_count.to_csv(f"{save_dir}top{TOP_N}_{safe_g}_daily_count.csv", encoding='utf-8-sig')
    wide_pct.round(3).to_csv(f"{save_dir}top{TOP_N}_{safe_g}_daily_pct_day.csv", encoding='utf-8-sig')

# Save long table
long_path = f"{save_dir}top{TOP_N}_group_day_note_count_pctday_long.csv"
gdn.to_csv(long_path, index=False, encoding='utf-8-sig')

print("\nSaved files:")
print(f"- {long_path}")
print(f"- {save_dir}top{TOP_N}_<Group>_daily_count.csv")
print(f"- {save_dir}top{TOP_N}_<Group>_daily_pct_day.csv")

# =========================
# 6. Heatmap: Difference (B - A) based on Pct_Day
# =========================
# Double-check that these exact strings match your Group_Label data!
pct_A = make_wide(gdn, 'Group A (Mild/Stable)', 'Pct_Day')
pct_B = make_wide(gdn, 'Group B (Severe/Complex)', 'Pct_Day')

diff = pct_B - pct_A

# Sort by mean difference (Negative = Group A dominant, Positive = Group B dominant)
diff_sorted = diff.assign(sort_val=diff.mean(axis=1)).sort_values('sort_val').drop(columns='sort_val')

# Generate X-axis labels dynamically
x_labels = [f"D{d}" if d < 0 else "OP" if d == 0 else f"POD{d}" for d in all_days]

# Setup colormap and scale
cmap = LinearSegmentedColormap.from_list('custom', ['#3498db', '#ffffff', '#e74c3c'], N=256)
max_val = max(float(np.nanmax(np.abs(diff_sorted.values))), 1.0)

plt.figure(figsize=(16, 10))
sns.heatmap(
    diff_sorted, cmap=cmap, center=0,
    vmin=-max_val, vmax=max_val,
    annot=True, fmt='.1f', annot_kws={'size': 9},
    linewidths=0.5, linecolor='lightgray',
    cbar_kws={'label': 'Difference (% points) = Group B - Group A\n← A dominant | B dominant →'}
)

plt.title(f'Difference in Top {TOP_N} Nursing Notes (Pct_Day, B - A)', fontsize=16, fontweight='bold', pad=12)
plt.xlabel('Clinical Course')
plt.ylabel(f'Top {TOP_N} Nursing Notes')
plt.xticks(np.arange(len(all_days)) + 0.5, x_labels, rotation=0)
plt.tight_layout()

out_path = f"{save_dir}heatmap_top{TOP_N}_diff_pctDAY.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Heatmap saved: {out_path}")
