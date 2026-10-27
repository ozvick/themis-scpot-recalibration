# Recalibrating the spacecraft-potential density relation (THEMIS)

Analysis code for a study of the analytic spacecraft-potential-to-electron-density
conversion (`scpot2dens`, distributed in SPEDAS/PySPEDAS), evaluated against
machine-learning regression using THEMIS data.

**Status:** manuscript in preparation.

## Summary of findings

- `scpot2dens` has a domain-of-validity limit: the relation diverges as spacecraft
  potential approaches zero, producing unphysical densities during perigee passes
  (12.8% of samples, median radial distance 3.4 R_E). Usable range: V_sc >= 3.0 V.
- Within that domain the formula carries a two-parameter calibration error, with a
  log-space slope of 0.43 +/- 0.07 across days and 0.45 +/- 0.03 across probes.
- Direct ML regression does not recover the relation under leave-one-day-out
  validation (median R2 = -0.04).
- A hybrid estimator (recalibrated analytic + learned residual) performs best:
  median R2 = 0.506, the only approach without a catastrophic fold.

| Estimator | Median R2 | Worst R2 | Median MAE (dex) |
| --- | --- | --- | --- |
| scpot2dens, as published | -5.838 | -16.186 | 0.966 |
| Random Forest, all features | -0.040 | -3.877 | 0.317 |
| Ridge, spacecraft potential only | -0.780 | -4.021 | 0.423 |
| scpot2dens, recalibrated (2 params) | 0.015 | -1.069 | 0.276 |
| Hybrid: recalibrated + RF residual | 0.506 | -0.210 | 0.192 |

Leave-one-day-out validation across four days spanning 2007-2009.

## Setup

    pip install -r requirements.txt
    jupyter lab

Run `notebooks/03_diagnostics.ipynb` top to bottom. The first cell downloads THEMIS
ESA, FGM and STATE data plus OMNI solar-wind parameters via PySPEDAS and caches the
result, so later runs skip the download. Data caches are gitignored; all inputs are
public and regenerate automatically.

## Known limitations

- 2007-07-20 yields negative R2 for every estimator. It precedes the January 2008
  first epoch of the THEMIS calibration coefficients (Archer et al. 2019).
- Probe A is a calibration outlier (R2 = -0.239), possibly related to EFI deployment
  timing early in the mission.
- Spacecraft potential and ESA density are not independent: THEMIS moment processing
  uses the EFI potential to correct particle energies before computing density.
  Absolute skill scores are inflated; comparisons between estimators remain valid.
- Cross-mission validation was attempted and is not currently feasible.

## Citation

Manuscript in preparation. Please contact before citing.

## License

MIT
