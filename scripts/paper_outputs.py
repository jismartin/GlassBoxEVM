"""Tables 4-8 and Figures 2-7 of the paper (Sections 5-7).

Usage (from the repository root):
    python scripts/paper_outputs.py

Inputs
------
--repo           GlassBoxEVM repository root (Section 5 benchmark results and analysis CSVs).
--aggregate      Confirmatory aggregate directory produced by
                 `python -m scripts.sections67 aggregate` (network_metrics.csv,
                 local_metrics.csv, decisions.csv, inference.csv, h2_influence.csv).
                 Default: <repo>/results67/confirmatory/aggregate.
--pilot          Pilot outputs at commit 9ec86e3 (networks/<id>/explain/<model>/pd_curves.csv.gz
                 and aggregate/network_metrics.csv); used for Figure 6 only.
                 Default: <repo>/results67/pilot/figure6_inputs.
--structure360   Exploratory structure measures for the 360 benchmark networks; used for Figure 4
                 and Table 6 (exploratory column). Default:
                 <repo>/model_evaluation/analysis/structure_measures_360.csv.
--out            Output directory.

All bootstrap intervals for Section 5 are read from the benchmark analysis files
(non-stratified network bootstrap). Figure 7 intervals use a network bootstrap with
2,000 resamples and seed 20261122.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

GREY = '#888888'
FLOOR = 1e-8
COL = {'en-36': '#0072B2', 'ebm-24': '#E69F00', 'fastai-41': '#009E73', 'ols': '#555555',
       'xgb-39': '#CC79A7', 'sv-0': '#56B4E9', 'rf-14': '#D55E00'}
SHORT = {'en-36': 'EN', 'ebm-24': 'EBM', 'fastai-41': 'fastai', 'xgb-39': 'XGBoost',
         'sv-0': 'SVR', 'rf-14': 'RF', 'ols': 'OLS'}
NAME = {'en-36': 'Elastic Net (EN-36)', 'ebm-24': 'EBM (EBM-24)', 'fastai-41': 'fastai MLP (fastai-41)',
        'xgb-39': 'XGBoost (XGB-39)', 'sv-0': 'SVR (SVR-0)', 'rf-14': 'Random Forest (RF-14)',
        'ols': 'OLS (reference)'}
ORDER = ['en-36', 'ebm-24', 'fastai-41', 'xgb-39', 'sv-0', 'rf-14', 'ols']
SEL = {'en': 36, 'ebm': 24, 'fastai': 41, 'xgb': 39, 'sv': 0, 'rf': 14}
KEY = {'en': 'en-36', 'ebm': 'ebm-24', 'fastai': 'fastai-41', 'xgb': 'xgb-39', 'sv': 'sv-0', 'rf': 'rf-14'}
SUBSETS = ['J30', 'J60', 'J90', 'J120']
FIG6_NETWORKS = ['j9040_3', 'j902_5', 'j3025_2']


def style():
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'legend.fontsize': 7, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 0.6, 'savefig.dpi': 300,
                         'lines.linewidth': 1.2, 'font.family': 'sans-serif'})


def ci(v, lo, hi, d):
    return f'{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def slope(x, y):
    return np.polyfit(np.asarray(x), np.asarray(y), 1)[0]


def common_slope(df, y):
    """Slope on C with subset-specific intercepts (the H1 model)."""
    X = pd.get_dummies(df.subset).astype(float)
    X['C'] = df.C.values
    return np.linalg.lstsq(X.values, np.asarray(y), rcond=None)[0][-1]


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- Section 5
def section5(repo, out):
    rows = []
    for m, lhs in SEL.items():
        for f in glob.glob(f'{repo}/model_evaluation/results/model_evaluation_{m}_lhs_{lhs}_*.csv'):
            d = pd.read_csv(f, usecols=['project', 'subset', 'RelMSE', 'time'])
            d['model'] = m
            rows.append(d)
    bench = pd.concat(rows)
    an = f'{repo}/model_evaluation/analysis/'
    ols = pd.read_csv(an + 'ols_relmse_360.csv')
    s6 = pd.read_csv(an + 'model_comparison_summary_all.csv').set_index('model')
    s7 = pd.read_csv(an + 'seven_model_summary_all.csv').set_index('model')
    pe = pd.read_csv(an + 'pairwise_effect_sizes_all.csv')
    E = bench.assign(model=bench.model.map(KEY))[['project', 'subset', 'model', 'RelMSE']]
    E = pd.concat([E, ols[['project', 'subset', 'RelMSE']].assign(model='ols')])

    # Table 4
    t = s7.loc[ORDER]
    T4 = pd.DataFrame({
        'Model': [NAME[m] for m in ORDER], 'n': t.n_projects.values,
        'Mean rank (6 families)': [f"{s6.loc[m, 'mean_rank']:.3f}" if m in s6.index else '—' for m in ORDER],
        'Wins (6 families)': [str(int(s6.loc[m, 'wins'])) if m in s6.index else '—' for m in ORDER],
        'Mean rank (7 models)': [ci(r.mean_rank, r.mean_rank_ci_low_nb, r.mean_rank_ci_high_nb, 3) for _, r in t.iterrows()],
        'Wins (7 models)': t.wins.astype(int).values,
        'Median RelMSE': [ci(r.median_RelMSE, r.median_RelMSE_ci_low_nb, r.median_RelMSE_ci_high_nb, 4) for _, r in t.iterrows()],
        'Mean RelMSE': [ci(r.mean_RelMSE, r.mean_RelMSE_ci_low_nb, r.mean_RelMSE_ci_high_nb, 4) for _, r in t.iterrows()],
        'P95 RelMSE': [ci(r.p95_RelMSE, r.p95_RelMSE_ci_low_nb, r.p95_RelMSE_ci_high_nb, 3) for _, r in t.iterrows()],
        'Max RelMSE': [f'{v:.3f}' for v in t.max_RelMSE]})
    T4.to_csv(os.path.join(out, 'table4_global_accuracy.csv'), index=False)

    # Table 5a: paired comparisons
    W = E.pivot_table(index='project', columns='model', values='RelMSE')
    rows = []
    for a, b in [('en-36', 'ebm-24'), ('en-36', 'fastai-41'), ('ebm-24', 'fastai-41'), ('ols', 'en-36')]:
        d = W[a] - W[b]
        r = pe[(pe.model_a == a) & (pe.model_b == b)]
        rinv = pe[(pe.model_a == b) & (pe.model_b == a)]
        if len(r):
            r = r.iloc[0]
            row = (ci(r.mean_delta_RelMSE_a_minus_b, r.mean_delta_ci_low, r.mean_delta_ci_high, 4),
                   ci(r.median_delta_RelMSE_a_minus_b, r.median_delta_ci_low, r.median_delta_ci_high, 5),
                   f'{r.win_rate_a:.1%} [{r.win_rate_a_ci_low:.1%}, {r.win_rate_a_ci_high:.1%}]',
                   int(round(r.tie_rate * 360)), f'{SHORT[a]} − {SHORT[b]}')
        elif len(rinv):
            r = rinv.iloc[0]
            row = (ci(-r.mean_delta_RelMSE_a_minus_b, -r.mean_delta_ci_high, -r.mean_delta_ci_low, 4),
                   ci(-r.median_delta_RelMSE_a_minus_b, -r.median_delta_ci_high, -r.median_delta_ci_low, 5),
                   f'{1 - r.win_rate_a - r.tie_rate:.1%} [{1 - r.win_rate_a_ci_high - r.tie_rate:.1%}, '
                   f'{1 - r.win_rate_a_ci_low - r.tie_rate:.1%}]',
                   int(round(r.tie_rate * 360)), f'{SHORT[a]} − {SHORT[b]}')
        else:  # OLS extension: no bootstrap file
            row = (f'{d.mean():.4f}', f'{d.median():.5f}', f'{(d < 0).mean():.1%}', int((d == 0).sum()),
                   f'{SHORT[a]} − {SHORT[b]} (extension)')
        rows.append({'Comparison (a − b)': row[4], 'n': 360, 'Mean difference [95% CI]': row[0],
                     'Median difference [95% CI]': row[1], 'a has lower RelMSE [95% CI]': row[2], 'Exact ties': row[3]})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'table5a_paired_comparisons.csv'), index=False)

    # Table 5b: subset accuracy
    S5 = (E[E.model.isin(['en-36', 'ebm-24', 'fastai-41', 'ols'])]
          .groupby(['subset', 'model']).RelMSE.agg(['median', 'mean']).unstack('model').reindex(SUBSETS))
    T5b = pd.DataFrame({'Set': SUBSETS})
    for m in ['en-36', 'ebm-24', 'fastai-41', 'ols']:
        T5b[f'{SHORT[m]} median / mean'] = [f"{S5.loc[s, ('median', m)]:.4f} / {S5.loc[s, ('mean', m)]:.4f}" for s in SUBSETS]
    T5b.to_csv(os.path.join(out, 'table5b_subset_accuracy.csv'), index=False)

    # Figure 2
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    rng = np.random.default_rng(1)
    for i, m in enumerate(ORDER):
        v = E[E.model == m].RelMSE.values
        below = v < FLOOR
        y = np.where(below, FLOOR, v)
        x = i + rng.uniform(-0.28, 0.28, len(v))
        ax.scatter(x[~below], y[~below], s=4, color=COL[m], alpha=0.35, lw=0, zorder=2)
        if below.any():
            ax.scatter(x[below], y[below], s=6, facecolors='none', edgecolors=COL[m], lw=0.5, alpha=0.6, zorder=2)
        ax.hlines(np.median(v), i - 0.36, i + 0.36, color='black', lw=1.4, zorder=4)
        ax.scatter([i], [v.mean()], marker='D', s=22, color='white', edgecolor='black', lw=0.9, zorder=5)
    ax.set_yscale('log')
    ax.set_ylim(FLOOR / 2.5, 1.2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT[m] for m in ORDER])
    ax.axhline(FLOOR, color=GREY, lw=0.5, ls=':', zorder=1)
    ax.set_ylabel('Test RelMSE per network (log scale)')
    ax.set_title('Interpretable models have the lowest typical error; fastai has the lowest mean and lightest tail', loc='left')
    ax.text(0.01, 0.98, '↓ lower = better', transform=ax.transAxes, ha='left', va='top', color=GREY, fontsize=7)
    h = [Line2D([], [], color='black', lw=1.4, label='median'),
         Line2D([], [], marker='D', ls='', markerfacecolor='white', markeredgecolor='black', label='mean'),
         Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor='#555555',
                label='below 10⁻⁸ (drawn at the floor)')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)
    save(fig, out, 'figure2_error_distributions.png')

    # Figure 3
    tq = bench.groupby('model').time.quantile([.25, .5, .75]).unstack()
    xo = ols.time_raw.median()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for m, k in {v: k for k, v in KEY.items()}.items():
        q = tq.loc[k]
        x = q[0.5]
        for ax, (y, lo, hi) in zip(axs, [(s7.loc[m, 'median_RelMSE'], s7.loc[m, 'median_RelMSE_ci_low_nb'], s7.loc[m, 'median_RelMSE_ci_high_nb']),
                                         (s7.loc[m, 'mean_rank'], s7.loc[m, 'mean_rank_ci_low_nb'], s7.loc[m, 'mean_rank_ci_high_nb'])]):
            ax.errorbar(x, y, xerr=[[x - q[0.25]], [q[0.75] - x]], yerr=[[y - lo], [hi - y]], fmt='o',
                        color=COL[m], ms=5, lw=0.9, capsize=0)
            ax.annotate(SHORT[m], (x, y), xytext=(5, 4), textcoords='offset points', fontsize=7, color=COL[m])
    for ax, y in zip(axs, [s7.loc['ols', 'median_RelMSE'], s7.loc['ols', 'mean_rank']]):
        ax.scatter([xo], [y], marker='o', facecolors='none', edgecolors=COL['ols'], s=28, lw=0.9)
        ax.annotate('OLS*', (xo, y), xytext=(5, 4), textcoords='offset points', fontsize=7, color=COL['ols'])
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Median test RelMSE')
    axs[1].set_ylabel('Mean rank among seven models')
    for ax in axs:
        ax.set_xlabel('Training time per network (s, one core)')
    axs[1].invert_yaxis()
    axs[0].text(0.02, 0.98, '↓ lower = better', transform=axs[0].transAxes, va='top', fontsize=7, color=GREY)
    axs[1].text(0.02, 0.98, '↑ better rank', transform=axs[1].transAxes, va='top', fontsize=7, color=GREY)
    axs[0].set_title('a', loc='left', fontweight='bold')
    axs[1].set_title('b', loc='left', fontweight='bold')
    fig.suptitle('EN has the best mean rank at a median of 61 s; EBM matches its median error but takes about 12 times longer',
                 x=0.01, ha='left', fontsize=9)
    fig.text(0.01, -0.04, 'Points: median time (bars: interquartile range) and accuracy (bars: 95% bootstrap CI). '
             f'*OLS timed on a local laptop core (median {xo:.2f} s), not comparable with the cluster timings.',
             fontsize=6.5, color=GREY)
    fig.tight_layout()
    save(fig, out, 'figure3_accuracy_vs_time.png')


# ----------------------------------------------------------------------------- Section 6
def section6(agg, sm, out):
    nm = pd.read_csv(f'{agg}/network_metrics.csv')
    nm = nm[nm.cohort == 'confirmatory']
    inf = pd.read_csv(f'{agg}/inference.csv').set_index('statistic')
    hi = pd.read_csv(f'{agg}/h2_influence.csv')
    dd = {'EN − fastai': nm['RelMSE_EN-36'] - nm['RelMSE_fastai-41'],
          'EBM − fastai': nm['RelMSE_EBM-24'] - nm['RelMSE_fastai-41'],
          'EN − EBM': nm['RelMSE_EN-36'] - nm['RelMSE_EBM-24']}
    de = {'EN − fastai': sm['en-36'] - sm['fastai-41'], 'EBM − fastai': sm['ebm-24'] - sm['fastai-41'],
          'EN − EBM': sm['en-36'] - sm['ebm-24']}

    # Table 6
    rows = []
    for lab, st in [('EN − fastai', 'slope Delta_EN-fastai on C'), ('EBM − fastai', 'slope Delta_EBM-fastai on C')]:
        r = inf.loc[st]
        ss = [slope(nm.C[nm.subset == s], dd[lab][nm.subset == s]) for s in SUBSETS]
        rows.append({'Hypothesis': 'H1 (primary, IUT)', 'Quantity': f'Slope of RelMSE({lab}) on C',
                     'Estimate': f'{r.estimate:.4f}', '95% interval': f'[{r.ci_low:.4f}, {r.ci_high:.4f}]',
                     'Subset slopes J30/J60/J90/J120': ' / '.join(f'{v:.4f}' for v in ss),
                     'Exploratory 360 slope': f'{common_slope(sm, de[lab]):.4f}', 'Adjusted p': '—',
                     'Decision': 'Confirmed (both lower bounds > 0)'})
    r = inf.loc['beta_J30 - beta_J120 (Delta_EN-EBM on C)']
    ts = inf.loc['Theil-Sen (separate) J30 - J120']
    lab = 'EN − EBM'
    expl = (slope(sm.C[sm.subset == 'J30'], de[lab][sm.subset == 'J30'])
            - slope(sm.C[sm.subset == 'J120'], de[lab][sm.subset == 'J120']))
    rows.append({'Hypothesis': 'H2 (Holm family)', 'Quantity': 'β_J30 − β_J120, RelMSE(EN − EBM) on C',
                 'Estimate': f'{r.estimate:.4f}', '95% interval': f'[{r.ci_low:.4f}, {r.ci_high:.4f}]',
                 'Subset slopes J30/J60/J90/J120': ' / '.join(f"{inf.loc[f'beta_{s}', 'estimate']:.4f}" for s in SUBSETS),
                 'Exploratory 360 slope': f'{expl:.4f} (contrast)', 'Adjusted p': f'{r.p_holm:.3f}',
                 'Decision': 'Not confirmed'})
    rows.append({'Hypothesis': 'H2 sensitivity', 'Quantity': 'Theil–Sen J30 − J120 (separate fits)',
                 'Estimate': f'{ts.estimate:.4f}', '95% interval': f'[{ts.ci_low:.4f}, {ts.ci_high:.4f}]',
                 'Subset slopes J30/J60/J90/J120': '—', 'Exploratory 360 slope': '—', 'Adjusted p': '—',
                 'Decision': 'Descriptive; same sign'})
    mx = hi.loc[hi.loo_contrast_change.abs().idxmax()]
    rows.append({'Hypothesis': 'H2 influence', 'Quantity': "Cook's D > 4/n; max leave-one-out change in contrast",
                 'Estimate': f'{int(hi.flag_D_gt_4_over_n.sum())} networks; {mx.loo_contrast_change:.4f} ({mx.network_id})',
                 '95% interval': '—', 'Subset slopes J30/J60/J90/J120': '—', 'Exploratory 360 slope': '—',
                 'Adjusted p': '—', 'Decision': f'Max change ≈ {abs(mx.loo_contrast_change) / r.estimate:.0%} of estimate'})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'table6_structural_hypotheses.csv'), index=False)

    # Figure 4
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True)
    xs = np.array([0, nm.C.max()])
    for ax, (lab, m), letter in zip(axs[:2], [('EN − fastai', 'en-36'), ('EBM − fastai', 'ebm-24')], 'ab'):
        ax.scatter(sm.C, de[lab], s=7, facecolors='none', edgecolors=GREY, lw=0.4, alpha=0.6, zorder=1)
        ax.scatter(nm.C, dd[lab], s=6, color=COL[m], alpha=0.55, lw=0, zorder=2)
        b = common_slope(nm, dd[lab])
        a = np.mean(dd[lab]) - b * nm.C.mean()
        ax.plot(xs, a + b * xs, color='black', lw=1, zorder=3)
        ax.set_title(f'{letter}  RelMSE({lab})  [H1]', loc='left', fontsize=8)
        ax.text(0.03, 0.97, f'common slope {b:.3f}', transform=ax.transAxes, va='top', fontsize=7)
    ax = axs[2]
    lab = 'EN − EBM'
    ax.scatter(sm.C, de[lab], s=7, facecolors='none', edgecolors=GREY, lw=0.4, alpha=0.6, zorder=1,
               label='exploratory (360)')
    sc = {'J30': '#882255', 'J120': '#117733'}
    oth = ~nm.subset.isin(sc)
    ax.scatter(nm.C[oth], dd[lab][oth], s=5, color='#999999', alpha=0.5, lw=0, zorder=2, label='confirmatory J60, J90')
    for s, c in sc.items():
        k = nm.subset == s
        x, y = nm.C[k], dd[lab][k]
        b, a = np.polyfit(x, y, 1)
        ax.scatter(x, y, s=7, color=c, alpha=0.75, lw=0, zorder=3, label=f'confirmatory {s} (slope {b:.3f})')
        ax.plot([0, x.max()], [a, a + b * x.max()], color=c, lw=1.1, zorder=4)
    ax.set_title('c  RelMSE(EN − EBM)  [H2]', loc='left', fontsize=8)
    ax.legend(loc='lower left', frameon=False, fontsize=6, markerscale=1.4)
    for ax in axs:
        ax.axhline(0, color=GREY, lw=0.5, ls=':')
        ax.set_xlabel('Critical-set competition C')
    axs[0].set_ylabel('Paired difference in test RelMSE\n(below 0: first model better)')
    axs[0].set_xlim(-0.03, 0.8)
    fig.suptitle('Interpretable models lose their advantage over fastai as critical-set competition grows',
                 x=0.01, ha='left', fontsize=9)
    fig.tight_layout()
    save(fig, out, 'figure4_differences_vs_C.png')

    # Figure 5
    ab = {'EBM interactions\nRelMSE(EBM-noint) − RelMSE(EBM)': (nm['RelMSE_EBM-24-noint'] - nm['RelMSE_EBM-24'], 'ebm-24'),
          'EN pairwise products\nRelMSE(EN-main) − RelMSE(EN)': (nm['RelMSE_EN-36-main'] - nm['RelMSE_EN-36'], 'en-36'),
          'Uncentred products\nRelMSE(EN-centred) − RelMSE(EN)': (nm['RelMSE_EN-36-centred'] - nm['RelMSE_EN-36'], 'en-36')}
    fmt = lambda v: '≈ 0' if abs(v) < 5e-5 else f'{v:.4f}'
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True, sharey=True)
    for ax, (lab, (g, m)), letter in zip(axs, ab.items(), 'abc'):
        ax.scatter(nm.C, g, s=6, color=COL[m], alpha=0.55, lw=0)
        ax.axhline(0, color=GREY, lw=0.5, ls=':')
        ax.set_title(f'{letter}  {lab}', loc='left', fontsize=7.5)
        ax.set_xlabel('Critical-set competition C')
        ax.text(0.03, 0.98, f'mean gain  C = 0 (n = {(nm.C == 0).sum()}): {fmt(g[nm.C == 0].mean())}\n'
                f'                C > 0 (n = {(nm.C > 0).sum()}): {fmt(g[nm.C > 0].mean())}',
                transform=ax.transAxes, va='top', fontsize=6.3)
    axs[0].set_ylim(-0.012, 0.31)
    axs[0].set_ylabel('Gain in test RelMSE\n(above 0: full model better)')
    axs[0].set_xlim(-0.03, 0.8)
    fig.suptitle('Interaction and product terms improve accuracy only when critical sets compete (confirmatory, n = 320)',
                 x=0.01, ha='left', fontsize=9)
    fig.tight_layout()
    save(fig, out, 'figure5_ablation_gains_vs_C.png')
    return nm, inf


# ----------------------------------------------------------------------------- Section 7
def deriv(d, col):
    d = d.sort_values('grid_point')
    x, y = d.duration.values, d[col].values
    return (x[1:] + x[:-1]) / 2, np.diff(y) / np.diff(x)


def section7(agg, pilot, nm, inf, out):
    lm = pd.read_csv(f'{agg}/local_metrics.csv')
    L = lm[lm.cohort == 'confirmatory'].merge(nm[['network_id', 'C']], on='network_id')
    mods = ['EN-36', 'EBM-24', 'fastai-41', 'OLS', 'EN-36-main', 'EN-36-centred', 'EBM-24-noint']
    roles = ['tested', 'tested', 'reference', 'reference', 'ablation', 'ablation', 'ablation']
    rows = []
    for m, role in zip(mods, roles):
        x = L[L.model_id == m]
        rows.append({'Model': m, 'Networks': len(x), 'MAE_D mean (all)': f'{x.MAE_D.mean():.4f}',
                     'MAE_D median': f'{x.MAE_D.median():.4f}', 'MAE_D mean, C = 0': f'{x.MAE_D[x.C == 0].mean():.4f}',
                     'MAE_D mean, C > 0': f'{x.MAE_D[x.C > 0].mean():.4f}',
                     'Violation rate (tol 0.05), mean': f'{x.violation_rate.mean():.4f}',
                     'Violation excess, mean': f'{x.violation_excess_mean.mean():.5f}',
                     'Mean PD-curve error': f'{x.PD_error_mean.mean():.2e}', 'Role': role})
    r = inf.loc['mean EN-EBM MAE_D']
    rows.append({'Model': 'H3: EN-36 − EBM-24', 'Networks': 320,
                 'MAE_D mean (all)': ci(r.estimate, r.ci_low, r.ci_high, 4), 'MAE_D median': f"{r['median']:.4f}",
                 'Role': f'Holm p = {r.p_holm:.4f}; EN lower in {int(round(r.win_EN * 320))}/320; confirmed'})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'table7_local_fidelity.csv'), index=False)

    # Figure 6 (pilot networks)
    pn = pd.read_csv(f'{pilot}/aggregate/network_metrics.csv').set_index('network_id').C
    PD = {(n, m): pd.read_csv(f'{pilot}/networks/{n}/explain/{m}/pd_curves.csv.gz')
          for n in FIG6_NETWORKS for m in ['EN-36', 'EBM-24', 'fastai-41']}

    def pick(n):  # largest range of exact slope; ties broken by largest PD range
        st = []
        for a, g in PD[(n, 'EN-36')].groupby('activity_id'):
            _, dex = deriv(g, 'pd_exact')
            st.append((a, round(dex.max() - dex.min(), 6), g.pd_exact.max() - g.pd_exact.min()))
        st = pd.DataFrame(st, columns=['a', 'drange', 'pdrange']).sort_values(['drange', 'pdrange'], ascending=False)
        return int(st.a.iloc[0])

    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex='col')
    for j, n in enumerate(FIG6_NETWORKS):
        a = pick(n)
        e = PD[(n, 'EN-36')]
        e = e[e.activity_id == a].sort_values('grid_point')
        xm, dex = deriv(e, 'pd_exact')
        axs[0, j].plot(e.duration, e.pd_exact, color='black', lw=2, label='exact (simulation)', zorder=5)
        axs[1, j].plot(xm, dex, color='black', lw=2, zorder=5)
        for m, ls in [('EN-36', '-'), ('EBM-24', '-'), ('fastai-41', '--')]:
            d = PD[(n, m)]
            d = d[d.activity_id == a].sort_values('grid_point')
            xm2, dm = deriv(d, 'pd_model')
            axs[0, j].plot(d.duration, d.pd_model, color=COL[m.lower()], lw=1.1, ls=ls, label=SHORT[m.lower()], zorder=6)
            axs[1, j].plot(xm2, dm, color=COL[m.lower()], lw=1.1, ls=ls, zorder=6)
        axs[0, j].set_title(f'{n} (C = {pn[n]:.2f}), activity {a}', loc='left', fontsize=7.5)
        axs[1, j].set_xlabel('Activity duration')
        axs[1, j].set_ylim(-0.15, 1.25)
        for h in (0, 1):
            axs[1, j].axhline(h, color=GREY, lw=0.4, ls=':')
    axs[0, 0].set_ylabel('Partial dependence of\nproject duration')
    axs[1, 0].set_ylabel('Slope of partial dependence')
    axs[0, 0].legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.suptitle('EN partial dependence is linear by construction; EBM and fastai follow the exact curvature where critical sets compete',
                 x=0.01, ha='left', fontsize=9)
    fig.text(0.01, -0.045, 'Activity per network: largest range of the exact slope (ties broken by largest partial-dependence range), '
             'chosen without reference to model performance. Pilot outputs at commit 9ec86e3.\nThese curves average over scenarios; '
             'MAE_D (Table 7) compares scenario-level local effects, where EN pairwise products make the effect depend on the other durations.',
             fontsize=6.3, color=GREY)
    fig.tight_layout()
    save(fig, out, 'figure6_pd_curves.png')

    # Table 8 and Figure 7
    D = pd.read_csv(f'{agg}/decisions.csv')
    D = D[D.cohort == 'confirmatory']
    P = D.pivot_table(index=['network_id', 'k'], columns='policy', values='B_mean')
    P90 = D.pivot_table(index=['network_id', 'k'], columns='policy', values='B_p90')
    Cmap = nm.set_index('network_id').C
    lab = {'exact_greedy': 'Exact greedy', 'EN-36_greedy': 'EN greedy', 'EBM-24_greedy': 'EBM greedy',
           'fastai-41_greedy': 'fastai greedy', 'SSI_static': 'SSI static', 'CI_static': 'CI static'}
    rows = []
    for k in [1, 2, 3]:
        Pk, Qk = P.xs(k, level='k'), P90.xs(k, level='k')
        c = Cmap.reindex(Pk.index)
        for p in lab:
            loss = Pk.exact_greedy - Pk[p]
            rows.append({'k': k, 'policy': p, 'Policy': lab[p], 'Mean benefit': round(Pk[p].mean(), 4),
                         'Share of exact-greedy benefit': round(Pk[p].mean() / Pk.exact_greedy.mean(), 4),
                         'Mean P90 benefit': round(Qk[p].mean(), 4), 'Mean loss vs exact greedy': round(loss.mean(), 4),
                         f'Mean loss, C = 0 (n = {(c == 0).sum()})': round(loss[c == 0].mean(), 4),
                         f'Mean loss, C > 0 (n = {(c > 0).sum()})': round(loss[c > 0].mean(), 4),
                         'Networks equal to exact greedy': int((loss.abs() < 1e-12).sum())})
    DS = pd.DataFrame(rows)
    r4 = inf.loc['mean EN-EBM regret (k=3, duration units)']
    h4 = pd.DataFrame([{'k': 3, 'Policy': f'H4: regret EN − EBM = {r4.estimate:.4f} [{r4.ci_low:.4f}, {r4.ci_high:.4f}], '
                        f'Holm p = {r4.p_holm:.4f}; confirmed, difference favourable to EBM'}])
    pd.concat([DS.drop(columns='policy'), h4], ignore_index=True).to_csv(os.path.join(out, 'table8_decision_test.csv'), index=False)

    rng = np.random.default_rng(20261122)
    cols = {'EN-36_greedy': COL['en-36'], 'EBM-24_greedy': COL['ebm-24'], 'fastai-41_greedy': COL['fastai-41'],
            'SSI_static': '#555555', 'CI_static': '#AA4499'}
    off = {'EN-36_greedy': -0.2, 'EBM-24_greedy': -0.07, 'fastai-41_greedy': 0.07, 'SSI_static': 0.2}
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axs[0]
    for p, o in off.items():
        for k in [1, 2, 3]:
            Pk = P.xs(k, level='k')
            loss = (Pk.exact_greedy - Pk[p]).values
            bs = np.array([loss[rng.integers(0, len(loss), len(loss))].mean() for _ in range(2000)])
            mu = loss.mean()
            ax.errorbar(k + o, mu, yerr=[[mu - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - mu]], fmt='o',
                        ms=4, color=cols[p], lw=0.9, label=lab[p] if k == 1 else None)
    ax.axhline(0, color=GREY, lw=0.5, ls=':')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['k = 1', 'k = 2', 'k = 3'])
    ax.set_ylabel('Mean loss vs exact greedy\n(duration units)')
    ax.set_title('a  Loss in expected-duration reduction (95% CI)', loc='left', fontsize=8)
    ax.text(0.02, 0.97, '↓ lower = better', transform=ax.transAxes, va='top', fontsize=6.5, color=GREY)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax = axs[1]
    d3 = DS[DS.k == 3].set_index('policy')
    pp = ['EN-36_greedy', 'EBM-24_greedy', 'fastai-41_greedy', 'SSI_static', 'CI_static']
    for i, p in enumerate(pp):
        sh = d3.loc[p, 'Share of exact-greedy benefit'] * 100
        ax.barh(i, sh, color=cols[p], height=0.6)
        ax.text(sh + 1, i, f'{sh:.2f}%', va='center', fontsize=6.5)
    ax.set_yticks(range(len(pp)))
    ax.set_yticklabels([lab[p] for p in pp])
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.axvline(100, color=GREY, lw=0.5, ls=':')
    ax.set_xlabel('Share of exact-greedy mean benefit at k = 3 (%)')
    ax.set_title('b  Benefit retained (k = 3)', loc='left', fontsize=8)
    fig.suptitle('At k = 3, all surrogate-guided policies and the static SSI rule retain over 99.7% of the exact-greedy benefit',
                 x=0.01, ha='left', fontsize=9)
    cil = DS[DS.policy == 'CI_static']['Mean loss vs exact greedy']
    fig.text(0.01, -0.04, f'CI static omitted from panel a (mean loss {cil.min():.2f}–{cil.max():.2f}). '
             'Confirmatory cohort, n = 320; bootstrap over networks (2,000 resamples).', fontsize=6.3, color=GREY)
    fig.tight_layout()
    save(fig, out, 'figure7_decision_test.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='.')
    ap.add_argument('--aggregate', default=None)
    ap.add_argument('--pilot', default=None)
    ap.add_argument('--structure360', default=None)
    ap.add_argument('--out', default='paper_out')
    a = ap.parse_args()
    a.aggregate = a.aggregate or os.path.join(a.repo, 'results67/confirmatory/aggregate')
    a.pilot = a.pilot or os.path.join(a.repo, 'results67/pilot/figure6_inputs')
    os.makedirs(a.out, exist_ok=True)
    style()
    section5(a.repo, a.out)
    sm = pd.read_csv(a.structure360 or os.path.join(a.repo, 'model_evaluation/analysis/structure_measures_360.csv'))
    nm, inf = section6(a.aggregate, sm, a.out)
    section7(a.aggregate, a.pilot, nm, inf, a.out)


if __name__ == '__main__':
    main()
