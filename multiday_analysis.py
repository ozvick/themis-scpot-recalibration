# ── 9. Multi-day calibration: is the over-response stable across intervals? ─
# Reads every day cached by fetch_days.py and asks the question four days
# could not answer: across many independent intervals, is the high-branch
# calibration slope reliably below 1?
#
# With n=4 days the EIV upper bound exceeded 1 on two of them, so the pooled
# result was not supported day to day. This pools ~36 days instead.

import glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

CACHE_DIR = 'daycache'          # match fetch_days.py --cache-dir

files = sorted(glob.glob(os.path.join(CACHE_DIR, 'themis_*.pkl')))
print(f'{len(files)} cached days in {os.path.abspath(CACHE_DIR)}')
if not files:
    raise SystemExit('No cached days. Run fetch_days.py first.')

parts = []
for f in files:
    try:
        parts.append(pd.read_pickle(f))
    except Exception as e:
        print(f'  skip {os.path.basename(f)}: {e}')
big = pd.concat(parts, ignore_index=True)
print(f'pooled rows: {len(big)}   days: {big["day"].nunique()}   '
      f'probes: {sorted(big["probe"].unique())}')

# ── 9a. High-branch restriction (single continuous functional form) ────────
m = (big['ne'] > 0) & (big['ne_scpot'] > 0) & (big['scpot'] > 6.0)
hb = big[m].copy()
hb['log_ne'] = np.log10(hb['ne'])
hb['log_ne_analytic'] = np.log10(hb['ne_scpot'])
print(f'\nhigh branch (V_sc > 6 V): {len(hb)} rows '
      f'({100*len(hb)/len(big):.1f}% of pooled)')

def eiv(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = pearsonr(x, y)[0]
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    orth = ((sy**2 - sx**2) + np.sqrt((sy**2 - sx**2)**2 + 4*sxy**2))/(2*sxy)
    return dict(r=r, ols=r*sy/sx, orth=float(orth), sma=sy/sx,
                upper=sy/(r*sx), n=len(x))

# ── 9b. One calibration fit per day ────────────────────────────────────────
rows = []
for d in sorted(hb['day'].unique()):
    s = hb[hb['day'] == d]
    if len(s) < 200:
        continue
    e = eiv(s['log_ne_analytic'], s['log_ne'])
    rows.append({'day': d, 'n': e['n'], 'r': round(e['r'], 4),
                 'OLS': round(e['ols'], 4), 'orthogonal': round(e['orth'], 4),
                 'upper': round(e['upper'], 4),
                 'upper<1': e['upper'] < 1.0})
per_day = pd.DataFrame(rows)
print(f'\nPer-day high-branch calibration ({len(per_day)} days):\n')
print(per_day.to_string(index=False))

frac = per_day['upper<1'].mean()
print(f'\nDays with EIV upper bound < 1: {per_day["upper<1"].sum()}'
      f'/{len(per_day)} ({100*frac:.0f}%)')
for col in ['OLS', 'orthogonal', 'upper']:
    v = per_day[col]
    print(f'  {col:11s} median {v.median():.4f}   mean {v.mean():.4f} '
          f'+/- {v.std():.4f}   range [{v.min():.3f}, {v.max():.3f}]')

# ── 9c. Is the per-day slope distribution significantly below 1? ───────────
# Treat each DAY as the independent unit -- this respects the time correlation
# that a sample-level bootstrap ignores.
u = per_day['upper'].values
stat, p = wilcoxon(u - 1.0, alternative='less')
rng = np.random.default_rng(0)
bm = np.array([rng.choice(u, len(u), replace=True).mean() for _ in range(10000)])
lo, hi = np.percentile(bm, [2.5, 97.5])
print(f'\nDay-level test on the EIV upper bound (the hardest case):')
print(f'  Wilcoxon H1: median < 1  ->  p = {p:.4g}')
print(f'  mean upper bound {u.mean():.4f}, 95% CI (day bootstrap) '
      f'[{lo:.4f}, {hi:.4f}]')
if hi < 1.0:
    print('  -> Upper bound is below 1 even at the day level.')
    print(f'  -> Over-response is established: at least '
          f'{1/hi:.2f}x, likely ~{1/u.mean():.2f}x or more.')
else:
    print('  -> The day-level CI includes 1. The over-response is NOT')
    print('     established at the hardest assumption. Report the OLS/orthogonal')
    print('     range as the claim and state this limitation explicitly.')

uo = per_day['orthogonal'].values
bo = np.array([rng.choice(uo, len(uo), replace=True).mean() for _ in range(10000)])
lo_o, hi_o = np.percentile(bo, [2.5, 97.5])
print(f'\n  Orthogonal (equal-error) slope: mean {uo.mean():.4f}, '
      f'95% CI [{lo_o:.4f}, {hi_o:.4f}]')
print(f'  -> over-response {1/hi_o:.2f}x to {1/lo_o:.2f}x under that assumption')

# ── 9d. Does the slope drift with time? (calibration epochs) ───────────────
per_day['date'] = pd.to_datetime(per_day['day'])
if len(per_day) > 6:
    t = (per_day['date'] - per_day['date'].min()).dt.days.values.astype(float)
    for col in ['OLS', 'orthogonal']:
        sl = np.polyfit(t, per_day[col].values, 1)[0] * 365.25
        rr = pearsonr(t, per_day[col].values)
        print(f'\n{col} slope drift: {sl:+.4f} per year '
              f'(r = {rr[0]:+.3f}, p = {rr[1]:.3g})')
    print('A significant drift would support the published note that the')
    print('calibration coefficients change with time.')

# ── 9e. Figures ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

ax[0].hist(per_day['OLS'], bins=15, alpha=0.6, label='OLS', color='steelblue')
ax[0].hist(per_day['upper'], bins=15, alpha=0.6, label='EIV upper', color='crimson')
ax[0].axvline(1.0, color='k', ls='--', lw=1.5, label='no calibration error')
ax[0].set_xlabel('calibration slope a'); ax[0].set_ylabel('days')
ax[0].set_title('Per-day slope distribution'); ax[0].legend(fontsize=8)

ax[1].errorbar(per_day['date'], per_day['orthogonal'],
               yerr=[per_day['orthogonal']-per_day['OLS'],
                     per_day['upper']-per_day['orthogonal']],
               fmt='o', ms=4, capsize=2, color='steelblue', alpha=0.8)
ax[1].axhline(1.0, color='k', ls='--', lw=1.5)
ax[1].set_ylabel('slope (bars span OLS to EIV upper)')
ax[1].set_title('Slope vs time'); ax[1].tick_params(axis='x', rotation=45)

ax[2].scatter(per_day['n'], per_day['upper'], s=20, color='crimson')
ax[2].axhline(1.0, color='k', ls='--', lw=1.5)
ax[2].set_xlabel('samples that day'); ax[2].set_ylabel('EIV upper bound')
ax[2].set_title('Upper bound vs sample size')
plt.tight_layout(); plt.show()

# ── 9f. Per-probe, pooled over all days ────────────────────────────────────
rows = []
for pr in sorted(hb['probe'].unique()):
    s = hb[hb['probe'] == pr]
    if len(s) < 200: continue
    e = eiv(s['log_ne_analytic'], s['log_ne'])
    rows.append({'probe': pr, 'n': e['n'], 'r': round(e['r'], 4),
                 'OLS': round(e['ols'], 4), 'orthogonal': round(e['orth'], 4),
                 'upper': round(e['upper'], 4)})
print('\nPer-probe (pooled over all days):\n')
print(pd.DataFrame(rows).to_string(index=False))
