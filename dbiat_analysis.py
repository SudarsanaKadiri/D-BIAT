"""
dbiat_analysis.py
=================
Shared functions for the D-BIAT analysis notebooks.

    1. Settings and data loading
    2. Pre-processing (trial-level rules; check of the participant-level rules)
    3. Participant-level measures and D-scores
    4. Statistics (descriptives, t-tests, ANOVA, effect sizes, robust tests)
    5. ROC / AUC (bootstrap, DeLong, permutation, Youden, cross-validated classification)
    6. Output helpers (tables, figures)

Data: DBIAT_data_N135.csv (columns described in the README). Columns used here:
participant, group, task_version, block_number, block_type, trial_in_block, word_category, correct, rt_ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
MEASURES = ["life_rt", "death_rt", "life_rt_correct", "death_rt_correct",
            "life_er", "death_er", "D_RT", "D_ER", "D_Composite"]
GROUPS = ["Control", "Depressed", "Suicidal"]
CONTRASTS = [("Control", "Depressed"), ("Control", "Suicidal"), ("Depressed", "Suicidal")]
LIFE, DEATH = "Life:Me", "Death:Me"


# =====================================================================
# 1. Settings and data loading
# =====================================================================
def load_settings(path: str | Path | None = None) -> dict:
    import yaml
    with open(Path(path) if path else HERE / "settings.yaml") as fh:
        return yaml.safe_load(fh)


def path(cfg: dict, key: str, *parts) -> Path:
    """Absolute path: key is 'data' or an output subfolder ('tables', 'figures', 'derived')."""
    if key == "data":
        p = HERE / cfg["paths"]["data_file"]
        return p
    p = HERE / cfg["paths"]["outputs"] / key
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts)


def load_data(cfg: dict) -> pd.DataFrame:
    """All 120 trials of each of the 135 analyzed participants."""
    return pd.read_csv(path(cfg, "data"))


# =====================================================================
# 2. Pre-processing
# =====================================================================
@dataclass
class Prep:
    trials: pd.DataFrame              # retained trials
    participants: pd.DataFrame        # per-participant checks and counts


def preprocess(trials: pd.DataFrame, pp: dict) -> Prep:
    """Pre-processing rules (settings.yaml -> preprocessing), in this order:
    1. remove the last trial of every block (114 trials remain);
    2. participant-level rules: more than subject_fast_rt_max_prop of RTs < subject_fast_rt_ms, or
       accuracy < subject_min_accuracy, leads to exclusion (all 135 participants in the data file meet them);
    3. remove trials with RT outside [trial_rt_min_ms, trial_rt_max_ms].
    """
    t = trials[trials.trial_in_block < trials.trial_in_block.max()] if pp["drop_last_trial_of_block"] else trials
    by = t.groupby("participant")
    part = pd.DataFrame({
        "group": by.group.first(),
        "task_version": by.task_version.first(),
        "n_trials": by.size(),
        "prop_fast": by.rt_ms.apply(lambda x: (x < pp["subject_fast_rt_ms"]).mean()),
        "accuracy": by.correct.mean(),
    })
    part["meets_criteria"] = ((part.prop_fast <= pp["subject_fast_rt_max_prop"])
                              & (part.accuracy >= pp["subject_min_accuracy"]))
    outside = (t.rt_ms < pp["trial_rt_min_ms"]) | (t.rt_ms > pp["trial_rt_max_ms"])
    part["n_outside_window"] = outside.groupby(t.participant).sum()
    part["pct_outside_window"] = 100 * part.n_outside_window / part.n_trials
    kept = t[~outside & t.participant.isin(part.index[part.meets_criteria])]
    part["n_retained"] = kept.groupby("participant").size()
    return Prep(kept.reset_index(drop=True), part.reset_index())


# =====================================================================
# 3. Participant-level measures and D-scores
# =====================================================================
def subject_measures(trials: pd.DataFrame) -> pd.DataFrame:
    """Six behavioral measures and three D-scores per participant (retained trials).

      life_rt / death_rt        mean RT of all retained trials in the block type
      life_rt_correct / ...     mean RT of correct retained trials
      life_er / death_er        1 - proportion correct of retained trials
      D_RT        = (mean RT Life:Me - mean RT Death:Me) / SD of all retained RTs (ddof = 1)
      D_ER        = ER Life:Me - ER Death:Me
      D_Composite = mean(RT / mean RT of the participant + error) over Life:Me trials
                    - the same mean over Death:Me trials
    """
    t = trials.copy()
    t["err"] = 1.0 - t["correct"].astype(float)
    by = t.groupby("participant")
    mu, sd = by.rt_ms.mean(), by.rt_ms.std(ddof=1)
    t["comp"] = t.rt_ms / t.participant.map(mu) + t.err
    agg = t.groupby(["participant", "block_type"]).agg(rt=("rt_ms", "mean"), er=("err", "mean"),
                                                        comp=("comp", "mean"))
    agg = agg.join(t[t.correct == 1].groupby(["participant", "block_type"]).rt_ms.mean().rename("rt_c"))
    agg = agg.unstack("block_type")
    out = by[["group", "task_version"]].first()
    out["life_rt"], out["death_rt"] = agg[("rt", LIFE)], agg[("rt", DEATH)]
    out["life_rt_correct"], out["death_rt_correct"] = agg[("rt_c", LIFE)], agg[("rt_c", DEATH)]
    out["life_er"], out["death_er"] = agg[("er", LIFE)], agg[("er", DEATH)]
    out["D_RT"] = (out.life_rt - out.death_rt) / sd
    out["D_ER"] = out.life_er - out.death_er
    out["D_Composite"] = agg[("comp", LIFE)] - agg[("comp", DEATH)]
    return out.reset_index()


# =====================================================================
# 4. Statistics
# =====================================================================
def describe(df, col, by="group"):
    g = df.groupby(by)[col]
    out = pd.DataFrame({"n": g.count(), "mean": g.mean(), "sd": g.std(ddof=1),
                        "median": g.median(), "min": g.min(), "max": g.max()})
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["ci95_halfwidth_t"] = stats.t.ppf(0.975, out["n"] - 1) * out["se"]
    return out.reindex([x for x in GROUPS if x in out.index])


def cohen_d(x, y):
    """Cohen's d for independent groups: (mean x - mean y) / pooled SD."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp


def hedges_g(x, y):
    nx, ny = len(x), len(y)
    g = (1 - 3 / (4 * (nx + ny) - 9)) * cohen_d(x, y)
    se = math.sqrt((nx + ny) / (nx * ny) + g ** 2 / (2 * (nx + ny)))
    return g, g - 1.96 * se, g + 1.96 * se


def two_group_tests(x, y) -> dict:
    """Student t, Welch t, Mann-Whitney U and effect sizes for x vs y."""
    x = np.asarray(pd.Series(x).dropna(), float)
    y = np.asarray(pd.Series(y).dropna(), float)
    t, p = stats.ttest_ind(x, y)
    tw, pw = stats.ttest_ind(x, y, equal_var=False)
    sx, sy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df_w = (sx + sy) ** 2 / (sx ** 2 / (len(x) - 1) + sy ** 2 / (len(y) - 1))
    u, pu = stats.mannwhitneyu(x, y, alternative="two-sided")
    g, glo, ghi = hedges_g(x, y)
    return dict(n1=len(x), n2=len(y), mean1=x.mean(), mean2=y.mean(), diff=x.mean() - y.mean(),
                t=t, df=len(x) + len(y) - 2, p=p, t_welch=tw, df_welch=df_w, p_welch=pw,
                U=u, p_mwu=pu, d=cohen_d(x, y), g=g, g_ci_low=glo, g_ci_high=ghi)


def paired_tests(a, b) -> dict:
    """Paired t-test, d_z, 95% CI of the mean difference, Wilcoxon signed-rank test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    n = len(diff)
    t, p = stats.ttest_rel(a, b)
    h = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n)
    w, pw = stats.wilcoxon(a, b)
    return dict(n=n, mean_a=a.mean(), sd_a=a.std(ddof=1), mean_b=b.mean(), sd_b=b.std(ddof=1),
                mean_diff=diff.mean(), ci_low=diff.mean() - h, ci_high=diff.mean() + h,
                t=t, df=n - 1, p=p, d_z=diff.mean() / diff.std(ddof=1), W=w, p_wilcoxon=pw)


def oneway_anova(df, col, by="group") -> dict:
    """One-way ANOVA (with eta^2, omega^2), Welch ANOVA, Kruskal-Wallis, Levene and Brown-Forsythe."""
    groups = [df.loc[df[by] == g, col].dropna().values for g in GROUPS]
    F, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ss_t = ((allv - allv.mean()) ** 2).sum()
    k, N = len(groups), len(allv)
    ms_w = (ss_t - ss_b) / (N - k)
    w = np.array([len(g) / g.var(ddof=1) for g in groups])
    m = np.array([g.mean() for g in groups])
    mw = (w * m).sum() / w.sum()
    A = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (np.array([len(g) for g in groups]) - 1)).sum()
    Fw = A / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    df2w = (k ** 2 - 1) / (3 * lam)
    H, pk = stats.kruskal(*groups)
    lev = stats.levene(*groups, center="mean")
    bf = stats.levene(*groups, center="median")
    return dict(F=F, df1=k - 1, df2=N - k, p=p, eta2=ss_b / ss_t,
                omega2=(ss_b - (k - 1) * ms_w) / (ss_t + ms_w),
                F_welch=Fw, df2_welch=df2w, p_welch=stats.f.sf(Fw, k - 1, df2w),
                H=H, p_kruskal=pk, levene_W=lev.statistic, levene_p=lev.pvalue,
                brown_forsythe_W=bf.statistic, brown_forsythe_p=bf.pvalue)


def tukey_hsd(df, col, by="group") -> pd.DataFrame:
    res = stats.tukey_hsd(*[df.loc[df[by] == g, col].dropna().values for g in GROUPS])
    ci = res.confidence_interval()
    return pd.DataFrame([dict(group1=GROUPS[i], group2=GROUPS[j], diff=res.statistic[i, j],
                              ci_low=ci.low[i, j], ci_high=ci.high[i, j], p_tukey=res.pvalue[i, j])
                         for i, j in [(0, 1), (0, 2), (1, 2)]])


def games_howell(df, col, by="group") -> pd.DataFrame:
    g = {k: df.loc[df[by] == k, col].dropna().values for k in GROUPS}
    rows = []
    for a, b in CONTRASTS:
        x, y = g[a], g[b]
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
        dfree = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        p = float(stats.studentized_range.sf(abs(t) * math.sqrt(2), len(GROUPS), dfree))
        rows.append(dict(group1=a, group2=b, p_games_howell=min(p, 1.0)))
    return pd.DataFrame(rows)


def perm_chi2(table, n_perm: int, seed: int):
    """Chi-square statistic with a Monte Carlo (permutation) p-value for an r x c table."""
    table = np.asarray(table, int)
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    rows = np.repeat(np.arange(table.shape[0]), table.sum(1))
    cols = np.repeat(np.arange(table.shape[1]), table.sum(0))
    rng = np.random.default_rng(seed)
    exp = np.outer(table.sum(1), table.sum(0)) / table.sum()
    hits = 0
    for _ in range(n_perm):
        obs = np.zeros_like(table)
        np.add.at(obs, (rows, rng.permutation(cols)), 1)
        hits += ((obs - exp) ** 2 / exp).sum() >= chi2 - 1e-12
    return chi2, (hits + 1) / (n_perm + 1)


def fmt_p(p) -> str:
    return "< .001" if p < 0.001 else f"{p:.3f}".replace("0.", ".", 1)


def p_eq(p, name="p") -> str:
    f = fmt_p(p)
    return f"{name} {f}" if f.startswith("<") else f"{name} = {f}"


# =====================================================================
# 5. ROC / AUC
# =====================================================================
def auc_score(y, s) -> float:
    """AUC = P(score of positive > score of negative) + 0.5 P(tie); raw score direction."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _placements(pos, neg):
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return cmp.mean(1), cmp.mean(0)


def delong_ci(y, s, level: float = 0.95) -> tuple[float, float, float]:
    """AUC with the DeLong et al. (1988) confidence interval."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    v10, v01 = _placements(s[y == 1], s[y == 0])
    auc = v10.mean()
    se = math.sqrt(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))
    z = stats.norm.ppf(0.5 + level / 2)
    return auc, auc - z * se, auc + z * se


def delong_test(y, s1, s2) -> dict:
    """DeLong et al. (1988) test for two correlated AUCs."""
    y = np.asarray(y, int)
    comps = [_placements(np.asarray(s, float)[y == 1], np.asarray(s, float)[y == 0]) for s in (s1, s2)]
    aucs = np.array([c[0].mean() for c in comps])
    V10, V01 = np.vstack([c[0] for c in comps]), np.vstack([c[1] for c in comps])
    S = np.cov(V10) / V10.shape[1] + np.cov(V01) / V01.shape[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(var) if var > 0 else np.nan
    return dict(auc1=aucs[0], auc2=aucs[1], diff=aucs[0] - aucs[1], z=z, p=2 * stats.norm.sf(abs(z)))


def bootstrap_auc(y, s, n_boot: int, seed: int) -> np.ndarray:
    """Stratified bootstrap distribution of the AUC."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        bp = rng.choice(ip, len(ip), replace=True)
        bn = rng.choice(ineg, len(ineg), replace=True)
        out[b] = auc_score(np.r_[np.ones(len(bp)), np.zeros(len(bn))], np.r_[s[bp], s[bn]])
    return out


def permutation_auc_p(y, s, n_perm: int, seed: int) -> float:
    """Two-sided permutation p-value for |AUC - 0.5|."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    obs = abs(auc_score(y, s) - 0.5)
    rng = np.random.default_rng(seed)
    hits = sum(abs(auc_score(rng.permutation(y), s) - 0.5) >= obs - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def youden(y, s) -> dict:
    """Threshold maximizing sensitivity + specificity - 1 (score >= threshold -> positive)."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    best = None
    for thr in np.unique(s):
        pred = s >= thr
        sens = (pred & (y == 1)).sum() / (y == 1).sum()
        spec = (~pred & (y == 0)).sum() / (y == 0).sum()
        if best is None or sens + spec - 1 > best["J"]:
            best = dict(threshold=thr, sensitivity=sens, specificity=spec, J=sens + spec - 1)
    return best


def ppv_npv(sens, spec, prev):
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    return ppv, npv


def cv_classification(y, s, folds: int, repeats: int, seed: int) -> dict:
    """Repeated stratified k-fold CV of a single-score rule: the direction and the Youden threshold
    are learned in the training folds and applied to the held-out fold."""
    from sklearn.model_selection import StratifiedKFold
    y, s = np.asarray(y, int), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    se_, sp_, ba_ = [], [], []
    for _ in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=int(rng.integers(1e9)))
        pred = np.zeros(len(y), bool)
        for tr, te in skf.split(s, y):
            sign = 1.0 if s[tr][y[tr] == 1].mean() >= s[tr][y[tr] == 0].mean() else -1.0
            pred[te] = sign * s[te] >= youden(y[tr], sign * s[tr])["threshold"]
        se = (pred & (y == 1)).sum() / (y == 1).sum()
        sp = (~pred & (y == 0)).sum() / (y == 0).sum()
        se_.append(se); sp_.append(sp); ba_.append((se + sp) / 2)
    out = {}
    for k, v in [("cv_sensitivity", se_), ("cv_specificity", sp_), ("cv_balanced_accuracy", ba_)]:
        out[k], out[k + "_p2_5"], out[k + "_p97_5"] = float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    return out


def binormal_sens(auc: float) -> float:
    """Sensitivity = specificity at the optimal threshold of an equal-variance binormal model."""
    return stats.norm.cdf(math.sqrt(2) * stats.norm.ppf(auc) / 2)


# =====================================================================
# 6. Output helpers
# =====================================================================
def write_table(df: pd.DataFrame, cfg: dict, name: str, index: bool = False):
    df.to_csv(path(cfg, "tables", f"{name}.csv"), index=index)


def save_fig(fig, cfg: dict, name: str, tiff: bool = True):
    """Save <name>.pdf, <name>.png and (optionally) a 600-dpi TIFF."""
    fig.savefig(path(cfg, "figures", f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(path(cfg, "figures", f"{name}.png"), bbox_inches="tight", dpi=200)
    if tiff:
        fig.savefig(path(cfg, "figures", f"{name}.tif"), bbox_inches="tight", dpi=cfg["figures"]["dpi_tiff"],
                    pil_kwargs={"compression": "tiff_lzw"})


def finalize_tiff(cfg: dict, name: str):
    """Flatten <name>.tif to RGB and scale it to the journal width (180 mm) at 600 dpi."""
    from PIL import Image
    p = path(cfg, "figures", f"{name}.tif")
    im = Image.open(p)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1] if im.mode in ("RGBA", "LA") else None)
        im = bg
    dpi = cfg["figures"]["dpi_tiff"]
    w = round(cfg["figures"]["submission_width_mm"] / 25.4 * dpi)
    im = im.resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)
    im.save(p, dpi=(dpi, dpi), compression="tiff_lzw")
