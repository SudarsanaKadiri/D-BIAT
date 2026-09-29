"""
additional_analyses.py
======================
Additional analyses of the D-BIAT study (Results 3.6-3.7, Tables 3-4, Supplementary Tables S9-S14).

    Section                                                           Output in the article
    1. Group differences adjusted for depressive severity (PHQ-9);    Results 3.6; Supplementary Tables S9, S10
       partial correlations with symptom scores
    2. Ordered (gradient) predictions of Hypotheses 1-3               Table 4; Supplementary Table S11
       (linear trend, Jonckheere-Terpstra)
    3. Properties of the three D-scores (components, reliability)    Methods; Supplementary Table S12
    4. Conventional Brief-IAT scoring algorithm (Nosek et al., 2014)  Results 3.6; Table 3; Supplementary Table S13
    5. Apparent and cross-validated classification results           Table 3
    6. Task context: block order, practice/fatigue, implementation    Results 3.6; Supplementary Table S14

All settings (seeds, numbers of resamples) come from settings.yaml. Run from this folder:
    python additional_analyses.py
Results are written to outputs/tables/ and printed.
The clinical scores (PHQ-9, SIS, GAD-7) are read from the de-identified participant file
(settings.yaml -> paths.participants_file), which is not public (see README).
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dbiat_analysis as A  # noqa: E402

cfg = A.load_settings()
S = cfg["stats"]
SEED = S["random_seed"]
LAB = cfg["measures"]
pd.set_option("display.width", 250, "display.max_columns", 40)


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# =====================================================================
# 0. Data, pre-processing, scores (identical to notebooks 01-02)
# =====================================================================
raw = A.load_data(cfg)
prep = A.preprocess(raw, cfg["preprocessing"])
scores = A.subject_measures(prep.trials)
ref = pd.read_csv(A.path(cfg, "derived", "scores.csv"))
assert np.allclose(scores.sort_values("participant")[A.MEASURES].values,
                   ref.sort_values("participant")[A.MEASURES].values), "scores differ from R1"
print("Scores reproduce the R1 values for all 135 participants.")

clin = pd.read_csv(HERE / cfg["paths"]["participants_file"])
clin = clin.rename(columns={"subject": "participant"})[
    ["participant", "phq9_scr_total", "phq9_scr_minus_item9", "phq9_scr_item9", "sis_scr_total",
     "phq9_rec_total", "phq9_rec_minus_item9", "gad7_rec_total"]]
df = scores.merge(clin, on="participant", how="left", validate="1:1")
assert df.phq9_scr_total.notna().all() and len(df) == 135
first = raw[raw.block_number == 1].groupby("participant").block_type.first().rename("first_block")
df = df.merge(first, on="participant")
df["suicidal"] = (df.group == "Suicidal").astype(int)
DS = ["D_RT", "D_ER", "D_Composite"]

# =====================================================================
# 1. Adjustment for depressive severity -> Results 3.6; Supplementary Tables S9 and S10
# =====================================================================
section("1. Group differences adjusted for depressive severity")
rng_tab = df.groupby("group")[["phq9_scr_total", "phq9_scr_minus_item9", "phq9_rec_minus_item9", "sis_scr_total"]] \
    .agg(["min", "max", "mean", "std"]).reindex(A.GROUPS).round(2)
print("PHQ-9 / SIS ranges by group (analytic sample):\n", rng_tab)
A.write_table(rng_tab.reset_index(), cfg, "clinical_score_ranges", index=False)

dvs = df[df.group.isin(["Depressed", "Suicidal"])].copy()
covsets = {"none": None,
           "PHQ-9 screening, without item 9": "phq9_scr_minus_item9",
           "PHQ-9 screening, total": "phq9_scr_total",
           "PHQ-9 lab session, without item 9": "phq9_rec_minus_item9"}
rows = []
for m in A.MEASURES:
    g0, g1 = dvs[dvs.suicidal == 0][m], dvs[dvs.suicidal == 1][m]
    sp = math.sqrt(((len(g0) - 1) * g0.var(ddof=1) + (len(g1) - 1) * g1.var(ddof=1)) / (len(g0) + len(g1) - 2))
    for lab, cov in covsets.items():
        d = dvs.dropna(subset=[cov]) if cov else dvs
        f = f"{m} ~ suicidal" + (f" + {cov}" if cov else "")
        fit = smf.ols(f, data=d).fit()
        b, (lo, hi) = fit.params["suicidal"], fit.conf_int().loc["suicidal"]
        rows.append(dict(measure=m, covariate=lab, n=int(fit.nobs), b_suicidal=b, ci_low=lo, ci_high=hi,
                         t=fit.tvalues["suicidal"], df=fit.df_resid, p=fit.pvalues["suicidal"],
                         d_adj=b / sp, d_adj_low=lo / sp, d_adj_high=hi / sp,
                         b_cov=fit.params.get(cov, np.nan) if cov else np.nan,
                         p_cov=fit.pvalues.get(cov, np.nan) if cov else np.nan))
sev = pd.DataFrame(rows)
A.write_table(sev, cfg, "severity_adjusted_depressed_vs_suicidal")
print(sev[sev.measure.isin(DS + ["life_er"])].round(3).to_string())

# Dimensional analyses: partial correlations
section("1b. Dimensional analyses (Spearman partial correlations)")


def partial_spearman(d, x, y, z):
    r = d[[x, y] + ([z] if z else [])].rank()
    if z:
        rx = r[x] - np.polyval(np.polyfit(r[z], r[x], 1), r[z])
        ry = r[y] - np.polyval(np.polyfit(r[z], r[y], 1), r[z])
        k = 1
    else:
        rx, ry, k = r[x], r[y], 0
    rho = np.corrcoef(rx, ry)[0, 1]
    n = len(d)
    t = rho * math.sqrt((n - 2 - k) / (1 - rho ** 2))
    return rho, 2 * stats.t.sf(abs(t), n - 2 - k)


rows = []
for sample, d in [("Depressed + Suicidal (n = 90)", dvs), ("All participants (n = 135)", df)]:
    for m in A.MEASURES:
        for x, z in [("sis_scr_total", None), ("sis_scr_total", "phq9_scr_minus_item9"),
                     ("phq9_scr_minus_item9", None), ("phq9_scr_minus_item9", "sis_scr_total")]:
            rho, p = partial_spearman(d, m, x, z)
            rows.append(dict(sample=sample, measure=m, variable=x, controlling_for=z or "none", rho=rho, p=p))
partcor = pd.DataFrame(rows)
A.write_table(partcor, cfg, "partial_correlations_symptoms")
print(partcor[partcor.measure.isin(DS + ["life_er"])].round(3).to_string())
print("\nSpearman SIS vs PHQ-9 without item 9, n = 90:",
      np.round(stats.spearmanr(dvs.sis_scr_total, dvs.phq9_scr_minus_item9), 4),
      "; n = 135:", np.round(stats.spearmanr(df.sis_scr_total, df.phq9_scr_minus_item9), 4))

# =====================================================================
# 2. Ordered (gradient) predictions of Hypotheses 1-3 -> Table 4; Supplementary Table S11
# =====================================================================
section("2. Linear trend contrasts and Jonckheere-Terpstra tests (order Control, Depressed, Suicidal)")


def linear_trend(d, m):
    g = [d.loc[d.group == k, m].values for k in A.GROUPS]
    n = np.array([len(x) for x in g]); mu = np.array([x.mean() for x in g])
    w = np.array([-1.0, 0.0, 1.0])
    mse = sum(((x - x.mean()) ** 2).sum() for x in g) / (n.sum() - 3)
    est = (w * mu).sum(); se = math.sqrt(mse * (w ** 2 / n).sum())
    t = est / se
    return est, t, int(n.sum() - 3), 2 * stats.t.sf(abs(t), n.sum() - 3)


def jonckheere(d, m, n_perm, seed):
    """Jonckheere-Terpstra statistic for the order Control < Depressed < Suicidal; two-sided permutation p
    (standardized by the permutation distribution)."""
    x = d[m].values; lab = d.group.map({k: i for i, k in enumerate(A.GROUPS)}).values

    def J(labels):
        s = 0.0
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            xa, xb = x[labels == a], x[labels == b]
            s += (xb[:, None] > xa[None, :]).sum() + 0.5 * (xb[:, None] == xa[None, :]).sum()
        return s
    obs = J(lab)
    rng = np.random.default_rng(seed)
    perm = np.array([J(rng.permutation(lab)) for _ in range(n_perm)])
    z = (obs - perm.mean()) / perm.std(ddof=1)
    p = (np.sum(np.abs(perm - perm.mean()) >= abs(obs - perm.mean()) - 1e-9) + 1) / (n_perm + 1)
    return obs, z, p


rows = []
for j, m in enumerate(A.MEASURES):
    est, t, dfree, p = linear_trend(df, m)
    J, z, pj = jonckheere(df, m, S["n_permutations"], SEED + 500 + j)
    aov = A.oneway_anova(df, m)
    means = df.groupby("group")[m].mean().reindex(A.GROUPS)
    rows.append(dict(measure=m, mean_control=means.iloc[0], mean_depressed=means.iloc[1], mean_suicidal=means.iloc[2],
                     ordered_C_D_S_increasing=bool(means.is_monotonic_increasing),
                     ordered_C_D_S_decreasing=bool(means.is_monotonic_decreasing),
                     F=aov["F"], p_anova=aov["p"], trend_estimate=est, trend_t=t, trend_df=dfree, p_trend=p,
                     JT=J, JT_z=z, p_JT=pj))
trend = pd.DataFrame(rows)
A.write_table(trend, cfg, "trend_tests")
print(trend.round(4).to_string())

# =====================================================================
# 3. D-score properties (components, reliability) -> Methods (composite score); Supplementary Table S12
# =====================================================================
section("3. Components and reliability of the D-scores")
t = prep.trials.copy()
t["err"] = 1 - t.correct.astype(float)
mu = t.groupby("participant").rt_ms.transform("mean")
t["rt_ratio"] = t.rt_ms / mu
comp = t.groupby(["participant", "block_type"])[["rt_ratio", "err"]].mean().unstack()
parts = pd.DataFrame({"rt_part": comp[("rt_ratio", A.LIFE)] - comp[("rt_ratio", A.DEATH)],
                      "er_part": comp[("err", A.LIFE)] - comp[("err", A.DEATH)]}).reset_index()
chk = scores.merge(parts, on="participant")
assert np.allclose(chk.rt_part + chk.er_part, chk.D_Composite)
var_rt, var_er = chk.rt_part.var(), chk.er_part.var()
cov = np.cov(chk.rt_part, chk.er_part)[0, 1]
print(f"D_Composite = RT part + ER part. SD RT part = {chk.rt_part.std():.4f}, SD ER part = {chk.er_part.std():.4f}; "
      f"r(RT part, ER part) = {np.corrcoef(chk.rt_part, chk.er_part)[0, 1]:.3f}")
print(f"Share of Var(D_Composite): RT part {var_rt / chk.D_Composite.var():.3f}, ER part {var_er / chk.D_Composite.var():.3f}, "
      f"2cov {2 * cov / chk.D_Composite.var():.3f}")
cors = scores[DS].corr().round(3)
print("Correlations among D-scores:\n", cors)
comp_tab = pd.DataFrame([dict(sd_rt_part=chk.rt_part.std(), sd_er_part=chk.er_part.std(),
                              r_parts=np.corrcoef(chk.rt_part, chk.er_part)[0, 1],
                              share_var_rt=var_rt / chk.D_Composite.var(), share_var_er=var_er / chk.D_Composite.var(),
                              share_2cov=2 * cov / chk.D_Composite.var(),
                              r_comp_drt=cors.loc["D_Composite", "D_RT"], r_comp_der=cors.loc["D_Composite", "D_ER"],
                              r_drt_der=cors.loc["D_RT", "D_ER"])])
A.write_table(comp_tab, cfg, "dscore_composite_components")


def split_half(trials, scorer):
    """Odd-even split-half reliability (Spearman-Brown): trials alternately assigned within each participant x block."""
    tt = trials.sort_values(["participant", "block_number", "trial_in_block"]).copy()
    tt["half"] = tt.groupby(["participant", "block_type"]).cumcount() % 2
    a = scorer(tt[tt.half == 0]); b = scorer(tt[tt.half == 1])
    r = np.corrcoef(a.values, b.reindex(a.index).values)[0, 1]
    return r, 2 * r / (1 + r)


def score_cols(tr):
    return A.subject_measures(tr).set_index("participant")


# Conventional Brief-IAT scoring (Nosek et al., 2014), defined here so that its reliability is computed alike
def biat_prepare(raw_trials, drop_last=False):
    t = raw_trials.copy()
    if drop_last:
        t = t[t.trial_in_block < t.trial_in_block.max()]
    t = t[t.rt_ms <= 10000]                       # 1. trials > 10,000 ms removed
    t = t[t.trial_in_block > 4]                   # 2. first four trials of each block discarded
    t = t.assign(rt=t.rt_ms.clip(400, 2000))      # 3. latencies < 400 -> 400, > 2000 -> 2000
    return t


def biat_D(t):
    """D for each pair of consecutive blocks (1-2, 3-4, 5-6) = (mean Life:Me - mean Death:Me) / SD of the
    pair's trials; the D-score is the mean over the three pairs (negative = faster in Life:Me)."""
    t = t.assign(pair=(t.block_number - 1) // 2)
    def one(g):
        m = g.groupby("block_type").rt.mean()
        return (m.get(A.LIFE, np.nan) - m.get(A.DEATH, np.nan)) / g.rt.std(ddof=1)
    Dp = t.groupby(["participant", "pair"]).apply(one, include_groups=False).unstack()
    return Dp


rel = []
for m in DS:
    r, sb = split_half(prep.trials, lambda tr, m=m: score_cols(tr)[m])
    rel.append(dict(score=m, method="odd-even split half, Spearman-Brown", r_half=r, reliability=sb))
Dp = biat_D(biat_prepare(raw))
k = Dp.shape[1]
alpha = k / (k - 1) * (1 - Dp.var(ddof=1).sum() / Dp.sum(1).var(ddof=1))
rel.append(dict(score="D_BIAT (conventional)", method="Cronbach's alpha over the three block-pair D-scores",
                r_half=np.nan, reliability=alpha))
tb = biat_prepare(raw)
r, sb = split_half(tb, lambda tr: biat_D(tr).mean(1))
rel.append(dict(score="D_BIAT (conventional)", method="odd-even split half, Spearman-Brown", r_half=r, reliability=sb))
rel = pd.DataFrame(rel)
A.write_table(rel, cfg, "dscore_reliability")
print(rel.round(3).to_string())

# =====================================================================
# 4. Conventional Brief-IAT scoring -> Results 3.6; Table 3 (D_BIAT rows); Supplementary Table S13
# =====================================================================
section("4. Conventional Brief-IAT scoring algorithm (Nosek et al., 2014)")
fast = raw.groupby("participant").rt_ms.apply(lambda x: (x < 300).mean())
acc = raw.groupby("participant").correct.mean()
print(f"Participants with > 10% of trials < 300 ms (all 120 trials): {(fast > .10).sum()}; accuracy < 70%: {(acc < .70).sum()}")
variants = {
    "D_BIAT": dict(drop_last=False, min_acc=None),
    "D_BIAT_acc70": dict(drop_last=False, min_acc=0.70),
    "D_BIAT_114": dict(drop_last=True, min_acc=None),
}
biat = pd.DataFrame(index=scores.set_index("participant").index)
for name, v in variants.items():
    Dp = biat_D(biat_prepare(raw, v["drop_last"]))
    s = Dp.mean(1)
    if v["min_acc"]:
        s = s[acc.reindex(s.index) >= v["min_acc"]]
    biat[name] = s
biat = biat.reset_index().merge(scores[["participant", "group", "D_RT", "D_Composite"]], on="participant")
biat.to_csv(A.path(cfg, "derived", "scores_conventional_BIAT.csv"), index=False)
n_trials = biat_prepare(raw).groupby("participant").size()
print(f"Trials per participant after conventional pre-processing: {n_trials.min()}-{n_trials.max()}; "
      f"recoded latencies: {((biat_prepare(raw).rt_ms < 400) | (biat_prepare(raw).rt_ms > 2000)).mean() * 100:.2f}%")
print("r(D_BIAT, D_RT) =", round(biat[["D_BIAT", "D_RT"]].corr().iloc[0, 1], 3),
      "; r(D_BIAT, D_Composite) =", round(biat[["D_BIAT", "D_Composite"]].corr().iloc[0, 1], 3))

rows_desc, rows_pw, rows_auc = [], [], []
for vi, name in enumerate(variants):
    d = biat.dropna(subset=[name])
    desc = A.describe(d, name)
    aov = A.oneway_anova(d, name)
    tk = A.tukey_hsd(d, name).set_index(["group1", "group2"])
    for g in A.GROUPS:
        x = d.loc[d.group == g, name]
        tt = stats.ttest_1samp(x, 0)
        rows_desc.append(dict(score=name, group=g, n=len(x), mean=x.mean(), sd=x.std(ddof=1),
                              ci_half=desc.loc[g, "ci95_halfwidth_t"], t_vs_0=tt.statistic, p_vs_0=tt.pvalue,
                              F=aov["F"], df1=aov["df1"], df2=aov["df2"], p_anova=aov["p"], eta2=aov["eta2"],
                              p_welch_anova=aov["p_welch"], p_kruskal=aov["p_kruskal"],
                              p_trend=linear_trend(d, name)[3]))
    for k_, (a, b) in enumerate(A.CONTRASTS):
        res = A.two_group_tests(d.loc[d.group == a, name], d.loc[d.group == b, name])
        rows_pw.append(dict(score=name, group1=a, group2=b, **res, p_tukey=tk.loc[(a, b), "p_tukey"]))
        sub = d[d.group.isin([a, b])]
        y, s = (sub.group == b).astype(int).values, sub[name].values
        auc = A.auc_score(y, s)
        sign = 1 if auc >= .5 else -1
        boot = A.bootstrap_auc(y, s, S["n_bootstrap"], SEED + 1000 + 10 * vi + k_)
        yd = A.youden(y, sign * s)
        cv = A.cv_classification(y, s, S["cv_folds"], S["cv_repeats"], SEED + 1000 + vi + k_)
        rows_auc.append(dict(score=name, contrast=f"{a} vs {b}", n_ref=int((y == 0).sum()), n_pos=int((y == 1).sum()),
                             auc=auc, boot_ci_low=np.percentile(boot, 2.5), boot_ci_high=np.percentile(boot, 97.5),
                             p_perm=A.permutation_auc_p(y, s, S["n_permutations"], SEED + 1100 + 10 * vi + k_),
                             sensitivity=yd["sensitivity"], specificity=yd["specificity"],
                             balanced_accuracy=(yd["sensitivity"] + yd["specificity"]) / 2, **cv))
biat_desc, biat_pw, biat_auc = pd.DataFrame(rows_desc), pd.DataFrame(rows_pw), pd.DataFrame(rows_auc)
A.write_table(biat_desc, cfg, "conventional_biat_descriptives_anova")
A.write_table(biat_pw, cfg, "conventional_biat_pairwise")
A.write_table(biat_auc, cfg, "conventional_biat_auc")
print(biat_desc.round(3).to_string())
print(biat_pw[["score", "group1", "group2", "t", "df", "p", "p_welch", "p_mwu", "d", "p_tukey"]].round(3).to_string())
print(biat_auc[["score", "contrast", "auc", "boot_ci_low", "boot_ci_high", "p_perm", "sensitivity", "specificity",
               "cv_balanced_accuracy", "cv_balanced_accuracy_p2_5", "cv_balanced_accuracy_p97_5"]].round(3).to_string())

# DeLong: D_BIAT vs D_RT and vs D_Composite (Control vs Suicidal and all contrasts)
rows = []
for a, b in A.CONTRASTS:
    sub = biat[biat.group.isin([a, b])]
    y = (sub.group == b).astype(int).values
    for other in ["D_RT", "D_Composite"]:
        s1, s2 = sub.D_BIAT.values, sub[other].values
        s1 = s1 if A.auc_score(y, s1) >= .5 else -s1
        s2 = s2 if A.auc_score(y, s2) >= .5 else -s2
        rows.append(dict(contrast=f"{a} vs {b}", comparison=f"D_BIAT vs {other}", **A.delong_test(y, s1, s2)))
biat_delong = pd.DataFrame(rows)
A.write_table(biat_delong, cfg, "conventional_biat_delong")
print(biat_delong.round(3).to_string())

# =====================================================================
# 5. Apparent and cross-validated classification results -> Table 3
# =====================================================================
section("5. Apparent vs cross-validated performance")
# The apparent (resubstitution) and cross-validated values were computed in notebook 05 (auc_full.csv, repeated
# stratified 5-fold CV, 200 repetitions; the score direction and the Youden threshold are learned in the training
# folds). They are assembled here in the layout of the revised Table 3.


aucfull = pd.read_csv(A.path(cfg, "tables", "auc_full.csv"))
rows = []
for k_, (a, b) in enumerate(A.CONTRASTS):
    sub = scores[scores.group.isin([a, b])]
    y = (sub.group == b).astype(int).values
    for j, m in enumerate(A.MEASURES):
        r = aucfull.query("contrast == @a + ' vs ' + @b and measure == @m").iloc[0]
        rows.append(dict(contrast=f"{a} vs {b}", measure=m, auc=r.auc, auc_aligned=r.auc_aligned,
                         boot_ci_low=r.boot_ci_low, boot_ci_high=r.boot_ci_high, p_perm=r["p_perm_vs_0.5"],
                         sensitivity=r.sensitivity, specificity=r.specificity,
                         balanced_accuracy=r.balanced_accuracy_resub,
                         cv_sensitivity=r.cv_sensitivity, cv_specificity=r.cv_specificity,
                         cv_balanced_accuracy=r.cv_balanced_accuracy,
                         cv_balanced_accuracy_p2_5=r.cv_balanced_accuracy_p2_5,
                         cv_balanced_accuracy_p97_5=r.cv_balanced_accuracy_p97_5))
table3 = pd.DataFrame(rows)
A.write_table(table3, cfg, "Table3_apparent_and_cv")
print(table3.round(3).to_string())

# =====================================================================
# 6. Task context: block order, practice/fatigue, implementation -> Results 3.6; Supplementary Table S14
# =====================================================================
section("6. Block order, change across blocks, task implementation")
tab = pd.crosstab(df.group, df.first_block).reindex(A.GROUPS)
chi2, p = A.perm_chi2(tab.values, S["n_chi2_permutations"], SEED)
print(tab, f"\nFirst block by group: chi2(2) = {chi2:.2f}, Monte Carlo p = {p:.3f}")
A.write_table(tab.reset_index().assign(chi2=chi2, p_monte_carlo=p), cfg, "block_order_by_group")
rows = []
for m in A.MEASURES + ["D_BIAT"]:
    d = df.merge(biat[["participant", "D_BIAT"]], on="participant")
    for adj in [False, True]:
        f = smf.ols(f"{m} ~ C(group, Treatment('Control'))" + (" + C(first_block)" if adj else ""), data=d).fit()
        rows.append(dict(measure=m, adjusted_for_first_block=adj,
                         b_CS=f.params["C(group, Treatment('Control'))[T.Suicidal]"],
                         p_CS=f.pvalues["C(group, Treatment('Control'))[T.Suicidal]"],
                         b_CD=f.params["C(group, Treatment('Control'))[T.Depressed]"],
                         p_CD=f.pvalues["C(group, Treatment('Control'))[T.Depressed]"],
                         b_first_LifeMe=f.params.get("C(first_block)[T.Life:Me]", np.nan),
                         p_first=f.pvalues.get("C(first_block)[T.Life:Me]", np.nan)))
order_adj = pd.DataFrame(rows)
A.write_table(order_adj, cfg, "block_order_adjusted_contrasts")
print(order_adj.round(3).to_string())
rows = []
for m in ["D_RT", "D_Composite"]:
    fit = smf.ols(f"{m} ~ C(group) * C(first_block)", data=df).fit()
    import statsmodels.api as sm
    tab = sm.stats.anova_lm(fit, typ=2)
    for eff in tab.index[:-1]:
        rows.append(dict(measure=m, model="group x first block", effect=eff, F=tab.loc[eff, "F"],
                         df1=tab.loc[eff, "df"], df2=tab.loc["Residual", "df"], p=tab.loc[eff, "PR(>F)"]))
    fit = smf.ols(f"{m} ~ C(group) + C(task_version)", data=df).fit()
    tab = sm.stats.anova_lm(fit, typ=2)
    for eff in tab.index[:-1]:
        rows.append(dict(measure=m, model="group + task version", effect=eff, F=tab.loc[eff, "F"],
                         df1=tab.loc[eff, "df"], df2=tab.loc["Residual", "df"], p=tab.loc[eff, "PR(>F)"]))
context = pd.DataFrame(rows)
print(context.round(3).to_string())
print("Mean D by first block:\n", df.groupby("first_block")[["D_RT", "D_Composite"]].agg(["mean", "std", "count"]).round(3))

# change across the six blocks (practice / fatigue): mean RT and error rate by block position
tb = prep.trials.copy(); tb["err"] = 1 - tb.correct
blk = tb.groupby(["participant", "block_number"]).agg(rt=("rt_ms", "mean"), er=("err", "mean")).reset_index()
blk = blk.merge(scores[["participant", "group"]], on="participant")
slopes = blk.groupby("participant").apply(lambda g: pd.Series({
    "rt_slope": np.polyfit(g.block_number, g.rt, 1)[0], "er_slope": np.polyfit(g.block_number, g.er, 1)[0]}),
    include_groups=False).reset_index().merge(scores[["participant", "group"]], on="participant")
for v in ["rt_slope", "er_slope"]:
    tt = stats.ttest_1samp(slopes[v], 0)
    aov = A.oneway_anova(slopes, v)
    rows.append(dict(measure=v, model="per-participant slope across blocks 1-6", effect=f"mean = {slopes[v].mean():.3f}",
                     F=aov["F"], df1=aov["df1"], df2=aov["df2"], p=aov["p"], t_vs_0=tt.statistic, p_vs_0=tt.pvalue))
    print(f"{v}: mean {slopes[v].mean():.3f} per block (t = {tt.statistic:.2f}, p = {tt.pvalue:.4f}); "
          f"group difference F = {aov['F']:.2f}, p = {aov['p']:.3f}; by group:",
          slopes.groupby('group')[v].mean().reindex(A.GROUPS).round(3).to_dict())
print(blk.groupby(["group", "block_number"]).rt.mean().unstack().reindex(A.GROUPS).round(1))
context = pd.DataFrame(rows)
A.write_table(context, cfg, "task_context")
print("\nDone.")
