#!/usr/bin/env python3
"""
fetch_days.py — multi-day THEMIS loader for the scpot2dens calibration study.

Downloads one day per month and caches each day separately so the run is
RESUMABLE: re-running skips days already on disk. Safe to kill and restart.

    python3 fetch_days.py                 # default: 2008-01 .. 2010-12 (36 days)
    python3 fetch_days.py --start 2008-01 --end 2009-12
    python3 fetch_days.py --day-of-month 15
    python3 fetch_days.py --cache-dir ./daycache

Expect hours of runtime and tens of GB of CDF files under ~/spedas_data.
Run it detached so it survives a closed terminal:

    nohup python3 fetch_days.py > fetch.log 2>&1 &
    tail -f fetch.log
"""
import argparse, os, sys, traceback
from datetime import datetime

import numpy as np
import pandas as pd

RE_KM = 6371.0
RESAMPLE = '30s'
PROBES = ['a', 'b', 'c', 'd', 'e']
OMNI_VARS = {'sw_bz': 'BZ_GSM', 'sw_speed': 'flow_speed',
             'sw_dens': 'proton_density', 'sw_pdyn': 'Pressure',
             'sym_h': 'SYM_H'}


def month_days(start, end, dom):
    """['2008-01', '2010-12'] -> ['2008-01-15', ...] one per month."""
    y0, m0 = (int(x) for x in start.split('-'))
    y1, m1 = (int(x) for x in end.split('-'))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f'{y:04d}-{m:02d}-{dom:02d}')
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def safe_interp(t_src, y_src, t_ref):
    if t_src is None or y_src is None or len(np.atleast_1d(t_src)) == 0:
        return np.full(len(t_ref), np.nan)
    y = np.asarray(y_src, dtype=float)
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.full(len(t_ref), np.nan)
    return np.interp(t_ref, np.asarray(t_src)[good], y[good],
                     left=np.nan, right=np.nan)


def load_one_day(day):
    """Return a DataFrame for one day across all probes, or None."""
    import pyspedas
    from pyspedas import get_data
    from pyspedas.projects.themis import scpot2dens

    TR = [f'{day}/00:00:00', f'{day}/23:59:59']

    # OMNI once per day
    try:
        pyspedas.projects.omni.data(trange=TR, datatype='1min', time_clip=True)
        omni = {k: get_data(v) for k, v in OMNI_VARS.items()}
    except Exception as e:
        print(f'    OMNI failed: {e}', flush=True)
        omni = {k: None for k in OMNI_VARS}

    frames = []
    for P in PROBES:
        try:
            pyspedas.projects.themis.esa(
                trange=TR, probe=P, level='l2', time_clip=True,
                varnames=[f'th{P}_peer_density', f'th{P}_peir_density',
                          f'th{P}_peer_sc_pot', f'th{P}_peer_avgtemp',
                          f'th{P}_peer_vthermal'])
            pyspedas.projects.themis.fgm(trange=TR, probe=P, level='l2',
                                         time_clip=True)
            pyspedas.projects.themis.state(trange=TR, probe=P, time_clip=True)

            ne_d = get_data(f'th{P}_peer_density')
            if ne_d is None:
                print(f'    th{P}: no density', flush=True)
                continue
            t_ref = ne_d.times
            ne = np.asarray(ne_d.y, dtype=float)

            def grab(nm):
                d = get_data(nm)
                return (safe_interp(d.times, d.y, t_ref) if d is not None
                        else np.full_like(ne, np.nan))

            scpot = grab(f'th{P}_peer_sc_pot')
            Te = grab(f'th{P}_peer_avgtemp')
            vth = grab(f'th{P}_peer_vthermal')
            ni = grab(f'th{P}_peir_density')

            try:
                b_d = get_data(f'th{P}_fgs_gsm')
                B = safe_interp(b_d.times, np.linalg.norm(b_d.y, axis=1), t_ref)
            except Exception:
                B = np.full_like(ne, np.nan)

            X = Y = Z = np.full_like(ne, np.nan)
            try:
                pos = get_data(f'th{P}_pos_gsm')
                if pos is not None and pos.y.ndim == 2:
                    X = safe_interp(pos.times, pos.y[:, 0] / RE_KM, t_ref)
                    Y = safe_interp(pos.times, pos.y[:, 1] / RE_KM, t_ref)
                    Z = safe_interp(pos.times, pos.y[:, 2] / RE_KM, t_ref)
            except Exception:
                pass
            R = np.sqrt(X**2 + Y**2 + Z**2)

            ne_sp = np.full_like(ne, np.nan)
            try:
                dd = scpot2dens(scpot, t_ref, Te, t_ref, ne, t_ref,
                                ni, t_ref, P)
                if dd is not None:
                    ne_sp = np.asarray(dd, dtype=float)
            except Exception as e:
                print(f'    th{P}: scpot2dens failed: {e}', flush=True)

            d = pd.DataFrame({'time': t_ref, 'ne': ne, 'ne_scpot': ne_sp,
                              'ni': ni, 'scpot': scpot, 'Te': Te, 'vth': vth,
                              'B': B, 'x_re': X, 'y_re': Y, 'z_re': Z,
                              'R_re': R})
            d['dt'] = pd.to_datetime(d['time'], unit='s')
            d = (d.set_index('dt').resample(RESAMPLE).median()
                  .dropna(subset=['ne']).reset_index())
            tt = d['dt'].astype('int64').values / 1e9
            for k, v in omni.items():
                d[k] = safe_interp(v.times, v.y, tt) if v is not None else np.nan
            d['probe'], d['day'] = P, day
            d.dropna(subset=['ne', 'scpot', 'Te', 'B'], inplace=True)
            if len(d):
                frames.append(d.reset_index(drop=True))
                print(f'    th{P}: {len(d)} pts', flush=True)
        except Exception as e:
            print(f'    th{P} failed: {e}', flush=True)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2008-01')
    ap.add_argument('--end', default='2010-12')
    ap.add_argument('--day-of-month', type=int, default=15)
    ap.add_argument('--cache-dir', default='daycache')
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    days = month_days(args.start, args.end, args.day_of_month)
    print(f'{len(days)} days requested: {days[0]} .. {days[-1]}', flush=True)

    done, failed, skipped = [], [], []
    for i, day in enumerate(days, 1):
        path = os.path.join(args.cache_dir, f'themis_{day}.pkl')
        if os.path.exists(path):
            print(f'[{i}/{len(days)}] {day}  cached, skipping', flush=True)
            skipped.append(day)
            continue
        print(f'[{i}/{len(days)}] {day}  {datetime.now():%H:%M:%S}', flush=True)
        try:
            d = load_one_day(day)
            if d is None or len(d) < 50:
                print('    -> no usable data', flush=True)
                failed.append(day)
                continue
            d.to_pickle(path)
            print(f'    -> saved {len(d)} rows', flush=True)
            done.append(day)
        except KeyboardInterrupt:
            print('\nInterrupted. Re-run to resume.', flush=True)
            sys.exit(1)
        except Exception:
            traceback.print_exc()
            failed.append(day)

    print(f'\nnew: {len(done)}   cached: {len(skipped)}   failed: {len(failed)}',
          flush=True)
    if failed:
        print('failed days:', failed, flush=True)
    print(f'\nCache dir: {os.path.abspath(args.cache_dir)}', flush=True)
    print('Next: run the analysis cell to pool these days.', flush=True)


if __name__ == '__main__':
    main()
