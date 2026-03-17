"""
Period-wise change in Nursing Care Focus
- Time window: D-3 ~ POD#7
- Outcome: Domain proportion within subject-day (Binomial GEE)
- Multiple testing correction applied (Bonferroni)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable
from statsmodels.stats.multitest import multipletests
import warnings

# Suppress warnings for cleaner output during GEE fitting
warnings.filterwarnings('ignore')

# ==========================================================
# 0) Configuration & Mapping
# ==========================================================

# TODO: Replace with your actual data path or keep it dynamic
DATA_PATH = "sample_nursing_record_data.csv" 

NANDA_DOMAIN_LABEL = {
    1: "Pain/Comfort",
    2: "Respiration/Airway",
    3: "Circulation/Perfusion",
    4: "Infection/Protection",
    5: "Elimination/Digestion",
    6: "Fluids/Metabolism",
    7: "Cognition/Emotion/Function"
}

# TODO: Define your specific mapping from Diagnosis text to Domain Number here
# Example: DX2NANDA_DOMAIN = {"Acute pain": 1, "Impaired gas exchange": 2, ...}
DX2NANDA_DOMAIN = {} 

PERIOD_ORDER = [
    "Preop (D-3 to D-1)",
    "Immediate postop (POD0-4)",
    "Late postop (POD5-7)"
]

# ==========================================================
# Helper Functions
# ==========================================================

def day_to_period(d):
    """Categorize postoperative days into defined periods."""
    if d in [-3, -2, -1]: return PERIOD_ORDER[0]
    if d in [0, 1, 2, 3, 4]: return PERIOD_ORDER[1]
    if d in [5, 6, 7]: return PERIOD_ORDER[2]
    return np.nan

def fit_domain_gee(dsub_fit, cov_type="AR1"):
    """Fit Binomial GEE model and perform joint Wald test for period effects."""
    cov_struct = Autoregressive() if cov_type == "AR1" else Exchangeable()
    
    # Initialize GEE model
    gee_model = smf.gee(
        formula="prop ~ C(period)",
        groups="subject_id",
        time="day_int",
        data=dsub_fit,
        family=Binomial(),
        cov_struct=cov_struct,
        freq_weights=dsub_fit["total_dx"]
    )
    
    # Fit the model
    fit = gee_model.fit()

    # Joint Wald test for period effects (excluding intercept)
    param_names = list(fit.params.index)
    period_terms = [p for p in param_names if p.startswith("C(period)[T.")]
    
    if not period_terms:
        return np.nan
        
    R = np.zeros((len(period_terms), len(param_names)))
    for r_idx, term in enumerate(period_terms):
        R[r_idx, param_names.index(term)] = 1.0

    wtest = fit.wald_test(R, scalar=True)
    return float(np.asarray(wtest.pvalue).squeeze())


# ==========================================================
# Main Execution
# ==========================================================

def main():
    print("--- Starting Analysis ---")
    
    # -------------------------
    # 1) Load & Preprocess
    # -------------------------
    try:
        df_raw = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Please provide a valid path.")
        return

    df = df_raw[(df_raw["types"] == 2) & (df_raw["day_diff"].between(-3, 7))].copy()
    
    df["dx"] = df["note_text"].astype(str).str.strip()
    
    # IMPORTANT: Assumes DX2NANDA_DOMAIN is populated
    df["domain_no"] = df["dx"].map(DX2NANDA_DOMAIN)
    
    df = df.dropna(subset=["domain_no", "subject_id", "day_diff"]).copy()
    df["domain_no"] = df["domain_no"].astype(int)
    df["day_int"] = df["day_diff"].astype(int)

    # Assign periods
    df["period"] = df["day_int"].map(day_to_period)
    df = df.dropna(subset=["period"]).copy()
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)

    # -------------------------
    # 2) Build subject×day×domain grid
    # -------------------------
    DAYS = list(range(-3, 8))
    subjects = df[["subject_id"]].drop_duplicates()
    day_grid = pd.DataFrame({"day_int": DAYS})
    
    base = subjects.merge(day_grid, how="cross")
    base["period"] = base["day_int"].map(day_to_period)
    base = base.dropna(subset=["period"]).copy()
    base["period"] = pd.Categorical(base["period"], categories=PERIOD_ORDER, ordered=True)

    total_counts = df.groupby(["subject_id", "day_int"]).size().rename("total_dx").reset_index()
    domain_counts = df.groupby(["subject_id", "day_int", "domain_no"]).size().rename("count").reset_index()
    all_domains = pd.DataFrame({"domain_no": sorted(NANDA_DOMAIN_LABEL.keys())})

    long_df = (
        base.merge(all_domains, how="cross")
            .merge(total_counts, on=["subject_id", "day_int"], how="left")
            .merge(domain_counts, on=["subject_id", "day_int", "domain_no"], how="left")
    )

    long_df["total_dx"] = long_df["total_dx"].fillna(0).astype(int)
    long_df["count"] = long_df["count"].fillna(0).astype(int)
    
    # Safe division to calculate proportion
    long_df["prop"] = np.divide(
        long_df["count"], 
        long_df["total_dx"], 
        out=np.zeros_like(long_df["count"], dtype=float), 
        where=long_df["total_dx"] != 0
    )

    fit_df = long_df[long_df["total_dx"] > 0].copy()

    # -------------------------
    # 3) GEE Modeling per Domain
    # -------------------------
    results = []
    for dom_no in sorted(NANDA_DOMAIN_LABEL.keys()):
        dsub_fit = fit_df[fit_df["domain_no"] == dom_no].copy()
        n_subj = dsub_fit["subject_id"].nunique()

        if n_subj < 30:
            results.append({
                "domain_no": dom_no, "domain": NANDA_DOMAIN_LABEL[dom_no],
                "n_subject": int(n_subj), "p_period_joint": np.nan, 
                "cov_struct": np.nan, "status": "SKIP (low n)"
            })
            continue

        try:
            # Try AR(1) first
            p_joint = fit_domain_gee(dsub_fit, cov_type="AR1")
            cov_used = "AR1"
        except Exception:
            # Fallback to Exchangeable
            try:
                p_joint = fit_domain_gee(dsub_fit, cov_type="EXCH")
                cov_used = "EXCH"
            except Exception as e:
                 results.append({
                    "domain_no": dom_no, "domain": NANDA_DOMAIN_LABEL[dom_no],
                    "n_subject": int(n_subj), "p_period_joint": np.nan,
                    "cov_struct": "FAILED", "status": f"Error: {str(e)}"
                 })
                 continue

        results.append({
            "domain_no": dom_no, "domain": NANDA_DOMAIN_LABEL[dom_no],
            "n_subject": int(n_subj), "p_period_joint": p_joint, 
            "cov_struct": cov_used, "status": "OK"
        })

    res_df = pd.DataFrame(results)

    # -------------------------
    # 4) Multiple Testing (Bonferroni)
    # -------------------------
    pvals = res_df["p_period_joint"].to_numpy(dtype=float)
    mask = np.isfinite(pvals)

    res_df["p_bonf"] = np.nan
    res_df["sig_bonf_0.05"] = False

    if mask.sum() > 0:
        rej, p_adj, _, _ = multipletests(pvals[mask], alpha=0.05, method="bonferroni")
        res_df.loc[mask, "p_bonf"] = p_adj
        res_df.loc[mask, "sig_bonf_0.05"] = rej

    print("\n=== Period effect on Nursing Care Focus (Binomial GEE) ===")
    print(res_df.sort_values("domain_no").to_string(index=False))

    # -------------------------
    # 5) OBSERVED Summaries
    # -------------------------
    obs = long_df[long_df["total_dx"] > 0].copy()
    obs["domain"] = obs["domain_no"].map(NANDA_DOMAIN_LABEL)

    # Weighted
    obs_weighted = obs.groupby(["period","domain_no","domain"], observed=False).agg(
        sum_count=("count","sum"), sum_total=("total_dx","sum")
    ).reset_index()
    obs_weighted["obs_mean_prop_weighted"] = obs_weighted["sum_count"] / obs_weighted["sum_total"]
    
    wide_w = obs_weighted.pivot_table(
        index="period", columns="domain", values="obs_mean_prop_weighted", aggfunc="first"
    ).reindex(PERIOD_ORDER)

    # Unweighted
    obs_unweighted = obs.groupby(["period","domain_no","domain"], observed=False).agg(
        obs_mean_prop_unweighted=("prop","mean")
    ).reset_index()
    
    wide_u = obs_unweighted.pivot_table(
        index="period", columns="domain", values="obs_mean_prop_unweighted", aggfunc="first"
    ).reindex(PERIOD_ORDER)

    print("\n=== OBSERVED mean proportion table (WEIGHTED) ===")
    print(wide_w.round(3).to_string())

    print("\n=== OBSERVED mean proportion table (UNWEIGHTED) ===")
    print(wide_u.round(3).to_string())

if __name__ == "__main__":
    main()
