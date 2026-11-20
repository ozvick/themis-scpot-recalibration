# Recalibrating the spacecraft-potential density relation (THEMIS)

Analysis code for a study of the analytic spacecraft-potential-to-electron-density
conversion (`scpot2dens`, distributed in SPEDAS/PySPEDAS), evaluated against
machine-learning regression using THEMIS data.

**Status: manuscript under revision following advisor review (Nov 2026).**
Several numbers in the first draft have been superseded — see *Revision history*
below. Figures and the estimator comparison are being recomputed; treat anything
not listed under *Current findings* as provisional.

## The audited formula

`scpot2dens` (PySPEDAS 2.2.2, a port of IDL `THM_SCPOT2DENS.PRO`, Feuerstein 2009,
implementing McFadden's empirical relation) is **piecewise in spacecraft potential**:

For V_sc <= 6 V, with d = offset - V_sc and offset = 2.2 (Te <= 500 eV) or 2.0:

    n = 460*10^(d/1.5) + 34*10^(d/0.7) + 1.6*10^(d/30)

For V_sc > 6 V:

    n = exp[(12 - V_sc) / (0.14*V_sc + 3.36)] / sqrt(Te)

The two branches are different functional forms and **do not meet**: the formula is
discontinuous by 0.8-1.4 dex at V_sc = 6 V (the size depends on Te). The low branch
has no explicit Te dependence; the high branch scales as 1/sqrt(Te).

All current analysis is restricted to the high branch (V_sc > 6 V) so that a single
continuous functional form is being calibrated.

## Current findings

**Calibration error, 22 days spanning 2008-2009 (191,075 samples, V_sc > 6 V):**

| statistic | value |
| --- | --- |
| days with EIV upper-bound slope < 1 | 19 / 22 |
| orthogonal slope (equal error variance) | 0.690, 95% CI [0.614, 0.763] |
| EIV upper bound (all error in predictor) | 0.833, 95% CI [0.763, 0.912] |
| Wilcoxon, day-level, H1: median < 1 | p = 5e-4 |
| **over-response factor** | **1.1x to 1.6x** |

Confidence intervals are day-level bootstraps, treating each day as the independent
unit, which respects the time correlation that a sample-level bootstrap ignores.

The three days where the upper bound exceeds 1 are among the smallest samples;
sample size predicts the upper bound across all days (r = -0.57, p = 0.006).
Restricting to days with n >= 6000 leaves 18 of 19 below 1.

**No detected calibration drift:** +0.043 per year, p = 0.46. Underpowered — the
span is 21 consecutive months of 2008-2009 plus one isolated 2010 day.

**Unphysical outputs at low potential:** the low branch grows steeply as V_sc
approaches zero. At V_sc = -18 V the 34*10^(d/0.7) term alone returns 2.2e30 cm^-3.
Negative potentials (eclipse, bias sweeps) must be screened before use.

## Revision history

| claim | first draft | after branch split | after 22 days |
| --- | --- | --- | --- |
| calibration slope | 0.43 +/- 0.07 | 0.58 (OLS) / 0.79 (EIV) | 0.63 +/- 0.15 per day |
| over-response | 2.2x | 1.3x - 1.7x | 1.1x - 1.6x |
| basis | 4 days, pooled over 6 V boundary | 4 days, high branch | 22 days, high branch |

Two corrections drove this:

1. **Errors-in-variables.** OLS underestimates the slope when the predictor carries
   measurement error. Reporting a family of estimators (OLS, orthogonal, SMA,
   inverse regression) bounds the effect instead of assuming all error is in the
   response.
2. **Simpson's paradox at the branch boundary.** The pooled slope (0.448) was lower
   than *either* branch's slope (0.595, 0.617), because the 6 V discontinuity shifts
   low-branch densities up by ~1 dex relative to the high branch. Pooling across the
   boundary biased the slope low.

## Repository contents

    notebooks/
      01_exploratory.ipynb     Initial multi-probe analysis
      02_generalization.ipynb  Leave-one-day-out and leave-one-probe-out validation
      03_diagnostics.ipynb     Domain limit, recalibration, hybrid estimator,
                               errors-in-variables slopes, formula branch analysis
      eiv_check.ipynb          Standalone errors-in-variables check (4 cells)
    fetch_days.py              Resumable multi-day loader (one day per month)
    multiday_analysis.py       Per-day calibration slopes + day-level bootstrap

## Reproducing

    pip install -r requirements.txt

Multi-day calibration (the current headline result):

    python3 fetch_days.py            # resumable; hours, ~8 GB of CDFs
    python3 multiday_analysis.py     # per-day slopes and day-level CI

`fetch_days.py` caches each day separately under `daycache/` and skips days already
present, so it can be interrupted and resumed. PySPEDAS caches raw CDFs under
`~/spedas_data`; those can be deleted afterwards without affecting the analysis,
since the per-day pickles hold everything needed.

Data caches are gitignored and not distributed. All inputs are public (THEMIS ESA,
FGM, STATE and OMNI via PySPEDAS from NASA/SPDF and the Berkeley THEMIS archive).

## Known limitations

- Restricted to V_sc > 6 V, which drops ~30% of samples. The excluded portion is
  not random: it is the lower-potential, higher-density end.
- Per-day slope scatter is large (OLS 0.63 +/- 0.15). The effect is established in
  aggregate; individual days vary considerably.
- Spacecraft potential and ESA density are not independent: THEMIS moment
  processing uses the EFI potential to correct particle energies before computing
  density. Absolute skill scores are inflated; comparisons between estimators
  remain valid, since the coupling affects all of them equally.
- Probe C has a bias-offset change at 2007-07-20 17:24 UT (2.2 -> 2.9, a +0.3 to
  +0.5 dex shift on the low branch). Other probes have a single epoch from
  2001-01-01 in the PySPEDAS source.
- Cross-mission validation was attempted and is not currently feasible: MMS
  regulates its potential (ASPOC, 0.09 V IQR), Cluster's indexed EFW files omit
  potential, and RBSP samples a non-overlapping potential range.

## Citation

Manuscript in preparation. Please contact before citing.

## License

MIT
