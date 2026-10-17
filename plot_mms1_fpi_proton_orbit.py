#!/usr/bin/env python3
"""
plot_mms1_fpi_proton_orbit.py
==============================
Visualizes Magnetospheric Multiscale (MMS1) FPI Fast Survey Ion Spectrogram alongside
spacecraft orbital parameters (Radial Distance R [R_E], GSE LT / MLT) over single-day or
multi-day time intervals.

Key Features:
  1. Multi-Day / Time-Range Support:
     - Accepts single date (`--date 2015-12-30`) or multi-day range (`--date 2015-12-30 2015-12-31`
       or `--trange 2015-12-30 2016-01-01`).
     - Defaults to 2015-12-30 and 2015-12-31 if no date is specified.
     - Smoothly concatenates multi-file intervals into a continuous time series.
  2. Coordinate Systems (GSE default, GSM optional via `--coord`):
     - GSE (default): Geocentric Solar Ecliptic, smooth orbital trajectory free of dipole wobble.
     - GSM: Geocentric Solar Magnetospheric, SM-based MLT reflecting geomagnetic boundary physics.
  3. Top Panel: Time-Energy Spectrogram with adaptive multi-parameter X-axis:
     - Date (for multi-day spans)
     - UTC (HH:MM)
     - Radial distance R [R_E]
     - GSE LT (or MLT in GSM mode)
  4. Bottom Panel: Radial Distance R [R_E] vs. Local Time Polar Orbit Trajectory:
     - Colored by ion differential energy flux at a selected energy band (~1.0 keV or ~5.0 keV).
     - Standard View: Sun / Noon (12 LT) at Top, Tail / Midnight (00 LT) at Bottom,
       Dawn (06 LT) at Right, Dusk (18 LT) at Left (configurable via `--noon top` or `--noon bottom`).
     - Earth boundary circle at R <= 1 R_E and radial distance rings up to MMS apogee (~12-13 R_E).
     - User-controllable point size (`--point-size`) for thin, razor-sharp orbit curves.
  4. Resilient Data Loading & Local Caching:
     - Reads local CDF files with `cdflib` for ultra-fast, robust data loading without pytplot crashes.
     - Automatically searches `./mms_data/`, `pydata/mms{probe}/`, or custom directories.
     - Automatic download via NASA SPDF / PySPEDAS (`spdf=True`) when files are missing locally.
     - Robust handling of fill values (`-1e31`, bad vectors, and instrument gaps).
  5. Fully headless execution (matplotlib 'Agg' backend) and high-resolution PNG output (300 DPI).

Author: Space Physics Pair-Programming Assistant
"""

import sys
import os
import glob
import argparse
import datetime
import urllib.request
import re

# Suppress harmless third-party warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cdflib
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors


def normalize_probe_name(probe_arg):
    """
    Normalizes probe input to string '1', '2', '3', or '4'.
    Supports integers 1..4, 'mms1'..'mms4', '1'..'4'.
    """
    if not probe_arg:
        return '1'
    p = str(probe_arg).strip().lower().replace('mms', '').replace('-', '')
    if p in ['1', '2', '3', '4']:
        return p
    return '1'


def parse_date_or_trange(date_args, trange_args=None):
    """
    Parses date or time range arguments into start_dt, end_dt, and a list of YYYY-MM-DD date strings.
    Supports formats:
      - 'YYYY-MM-DD', 'YYYYMMDD'
      - 'YYYY-MM-DD/HH:MM:SS', 'YYYY-MM-DD HH:MM:SS'
    """
    raw_args = trange_args if (trange_args and len(trange_args) > 0) else date_args
    if not raw_args:
        # Default to 2015-12-30 and 2015-12-31 (2 full days)
        raw_args = ['2015-12-30', '2015-12-31']

    if isinstance(raw_args, str):
        raw_args = raw_args.split()

    def parse_single_str(s):
        s = s.strip()
        for fmt in ('%Y-%m-%d/%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d/%H:%M', '%Y-%m-%d %H:%M',
                    '%Y-%m-%d', '%Y%m%d'):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                pass
        raise ValueError(f"Unrecognized date/time format: '{s}'")

    if len(raw_args) == 1:
        dt_start = parse_single_str(raw_args[0])
        if dt_start.hour == 0 and dt_start.minute == 0 and dt_start.second == 0:
            dt_end = dt_start + datetime.timedelta(days=1, microseconds=-1)
        else:
            dt_end = dt_start + datetime.timedelta(days=1)
    else:
        dt_start = parse_single_str(raw_args[0])
        dt_end = parse_single_str(raw_args[1])
        if dt_end.hour == 0 and dt_end.minute == 0 and dt_end.second == 0:
            dt_end = dt_end.replace(hour=23, minute=59, second=59, microsecond=999999)

    if dt_end < dt_start:
        dt_start, dt_end = dt_end, dt_start

    cur_date = dt_start.date()
    end_date = dt_end.date()
    calendar_dates = []
    while cur_date <= end_date:
        calendar_dates.append(cur_date.strftime('%Y-%m-%d'))
        cur_date += datetime.timedelta(days=1)

    return dt_start, dt_end, calendar_dates


def build_mms_file_index(search_dirs, probe='1'):
    """
    Performs a high-speed, single-pass directory scan over all search directories.
    Indexes files by date key (YYYYMMDD) in < 1 second.
    """
    fpi_index = {}
    mec_index = {}
    fpi_prefix = f'mms{probe}_fpi_fast_l2_dis-moms_'
    mec_prefix = f'mms{probe}_mec_srvy_l2_'

    seen_files = set()
    for sdir in search_dirs:
        norm_dir = os.path.normpath(sdir)
        if not os.path.exists(norm_dir):
            continue
        for root, _, files in os.walk(norm_dir):
            for fname in files:
                if not fname.endswith('.cdf') or fname in seen_files:
                    continue
                seen_files.add(fname)
                fpath = os.path.normpath(os.path.join(root, fname))

                if fname.startswith(fpi_prefix):
                    parts = fname[len(fpi_prefix):].split('_')
                    if parts and len(parts[0]) >= 8:
                        date_key = parts[0][:8]
                        fpi_index.setdefault(date_key, []).append(fpath)

                elif fname.startswith(mec_prefix):
                    m = re.search(r'(\d{8})', fname)
                    if m:
                        date_key = m.group(1)
                        mec_index.setdefault(date_key, []).append(fpath)

    for k in fpi_index:
        fpi_index[k].sort()
    for k in mec_index:
        mec_index[k].sort()

    return fpi_index, mec_index


def find_local_mms_files(calendar_dates, probe='1', data_dir=None, allow_download=False):
    """
    Rapidly locates local FPI and MEC CDF files using 1-pass directory indexing (< 1s).
    If allow_download is False and files are missing, reports missing dates immediately
    without wasting time querying remote servers.
    """
    search_dirs = []
    if data_dir:
        search_dirs.append(os.path.normpath(data_dir))
    search_dirs.extend([
        os.path.normpath(f'pydata/mms{probe}'),
        os.path.normpath(f'mms_data/mms{probe}'),
        os.path.normpath('mms_data'),
        os.path.normpath('pydata')
    ])

    print(f"[File Discovery] Rapidly indexing local directories (1-pass scan)...")
    t0 = datetime.datetime.now()
    fpi_index, mec_index = build_mms_file_index(search_dirs, probe=probe)
    dt_scan = (datetime.datetime.now() - t0).total_seconds()
    print(f"[File Discovery] Indexed in {dt_scan:.2f}s! Found {sum(len(v) for v in fpi_index.values())} FPI files, {sum(len(v) for v in mec_index.values())} MEC files.")

    fpi_files = []
    mec_files = []
    missing_dates_fpi = []
    missing_dates_mec = []

    for d_str in calendar_dates:
        date_compact = d_str.replace('-', '')

        if date_compact in fpi_index and len(fpi_index[date_compact]) > 0:
            fpi_files.extend(fpi_index[date_compact])
        else:
            missing_dates_fpi.append(d_str)

        if date_compact in mec_index and len(mec_index[date_compact]) > 0:
            mec_files.extend(mec_index[date_compact])
        else:
            missing_dates_mec.append(d_str)

    fpi_files = sorted(list(set(fpi_files)))
    mec_files = sorted(list(set(mec_files)))

    # If missing files detected
    if missing_dates_fpi or missing_dates_mec:
        if allow_download:
            print(f"[Downloader] --download specified: downloading missing dates (FPI: {len(missing_dates_fpi)}, MEC: {len(missing_dates_mec)})...")
            try:
                import pyspedas
                all_missing = sorted(list(set(missing_dates_fpi + missing_dates_mec)))
                t_start_dl = all_missing[0]
                t_end_dl = (datetime.datetime.strptime(all_missing[-1], '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                trange_dl = [t_start_dl, t_end_dl]

                if missing_dates_mec:
                    print(f"[Downloader] Fetching MEC ephemeris for {trange_dl}...")
                    pyspedas.mms_load_mec(trange=trange_dl, probe=probe, spdf=True, notplot=True, varformat='Epoch')

                if missing_dates_fpi:
                    print(f"[Downloader] Fetching FPI fast survey moments for {trange_dl}...")
                    pyspedas.mms_load_fpi(trange=trange_dl, probe=probe, datatype='dis-moms',
                                          data_rate='fast', level='l2', spdf=True, notplot=True, varformat='Epoch')

                # Re-index
                fpi_index, mec_index = build_mms_file_index(search_dirs, probe=probe)
                fpi_files = []
                mec_files = []
                for d_str in calendar_dates:
                    dc = d_str.replace('-', '')
                    if dc in fpi_index:
                        fpi_files.extend(fpi_index[dc])
                    if dc in mec_index:
                        mec_files.extend(mec_index[dc])
                fpi_files = sorted(list(set(fpi_files)))
                mec_files = sorted(list(set(mec_files)))
            except Exception as e:
                print(f"[Downloader] Download encountered an issue: {e}")
        else:
            print("\n" + "!" * 78)
            print(f"[Data Check Warning] Missing local CDF files for requested date range!")
            if missing_dates_fpi:
                n_miss = len(missing_dates_fpi)
                sample = missing_dates_fpi[:5]
                print(f"  - Missing FPI data ({n_miss} day{'s' if n_miss>1 else ''}): {sample}{' ...' if n_miss > 5 else ''}")
            if missing_dates_mec:
                n_miss = len(missing_dates_mec)
                sample = missing_dates_mec[:5]
                print(f"  - Missing MEC data ({n_miss} day{'s' if n_miss>1 else ''}): {sample}{' ...' if n_miss > 5 else ''}")
            print("\n[Recommendation] To download data fast and safely in background chunks, run:")
            print(f"  python download_mms_data.py --probe {probe} --trange {calendar_dates[0]} {calendar_dates[-1]}")
            print("Or add '--download' flag to this script to auto-download right now.")
            print("!" * 78 + "\n")

    if not fpi_files:
        raise FileNotFoundError(f"No MMS{probe} FPI dis-moms CDF files found locally for dates: {calendar_dates[:3]}...")
    if not mec_files:
        raise FileNotFoundError(f"No MMS{probe} MEC ephemeris CDF files found locally for dates: {calendar_dates[:3]}...")

    print(f"[File Discovery] Ready! Selected {len(fpi_files)} FPI files and {len(mec_files)} MEC files across {len(calendar_dates)} days.")
    return fpi_files, mec_files


def load_mms_fpi_and_orbit(fpi_files, mec_files, probe='1', coord='gse',
                           clip_start_dt=None, clip_end_dt=None):
    """
    Loads FPI Fast Survey ion flux and MEC orbital ephemeris.
    Interpolates R [R_E] and Local Time (GSE LT or GSM MLT) onto FPI timestamps.
    Returns cleaned data dictionary ready for plotting.
    """
    # -------------------------------------------------------------------------
    # 1. Parse FPI Ion Flux and Energy Data
    # -------------------------------------------------------------------------
    fpi_epochs_list = []
    fpi_flux_list = []
    fpi_energies = None

    spec_var_name = f'mms{probe}_dis_energyspectr_omni_fast'
    energy_var_name = f'mms{probe}_dis_energy_fast'

    print(f"[Parser] Reading {len(fpi_files)} FPI CDF files...")
    for f in fpi_files:
        try:
            cdf = cdflib.CDF(f)
            zvars = set(cdf.cdf_info().zVariables)

            # Resolve epoch
            ep_var = next((v for v in ['Epoch', 'Epoch_plus_var', 'epoch'] if v in zvars), None)
            if not ep_var:
                continue
            raw_ep = cdf.varget(ep_var)
            if raw_ep is None or np.size(raw_ep) == 0:
                continue

            u_sec = np.atleast_1d(cdflib.cdfepoch.unixtime(raw_ep))
            if u_sec.size == 0:
                continue

            # Resolve flux
            fl_var = next((v for v in [spec_var_name, 'mms1_dis_energyspectr_omni_fast'] if v in zvars), None)
            if not fl_var:
                continue
            fl_raw = cdf.varget(fl_var)
            if fl_raw is None or np.size(fl_raw) == 0:
                continue

            fl = np.asarray(fl_raw, dtype=np.float64)
            if fl.ndim == 1:
                fl = fl.reshape(1, -1)
            elif fl.ndim == 0 or fl.size == 0:
                continue

            # Resolve energy
            en_var = next((v for v in [energy_var_name, 'mms1_dis_energy_fast'] if v in zvars), None)
            if en_var and fpi_energies is None:
                raw_en = cdf.varget(en_var)
                if raw_en is not None and np.size(raw_en) > 0:
                    raw_en = np.asarray(raw_en, dtype=np.float64)
                    if raw_en.ndim == 2:
                        fpi_energies = raw_en[0]
                    elif raw_en.ndim == 1:
                        fpi_energies = raw_en

            # Validate record counts match
            if len(u_sec) != fl.shape[0]:
                print(f"[Parser] Warning: Skipping {os.path.basename(f)} due to mismatch between Epoch ({len(u_sec)}) and Flux ({fl.shape[0]}).")
                continue

            if fpi_energies is not None and fl.shape[1] != len(fpi_energies):
                print(f"[Parser] Warning: Skipping {os.path.basename(f)} due to energy bins mismatch ({fl.shape[1]} vs {len(fpi_energies)}).")
                continue

            fpi_epochs_list.append(u_sec)
            fpi_flux_list.append(fl)
        except Exception as e:
            print(f"[Parser] Warning: Skipping corrupted or unreadable FPI file {f}: {e}")

    if not fpi_epochs_list:
        raise ValueError("Could not extract any valid FPI data from provided CDF files.")

    all_fpi_epoch = np.concatenate(fpi_epochs_list)
    all_fpi_flux = np.concatenate(fpi_flux_list, axis=0)

    # Sort in chronological order if needed
    sort_idx = np.argsort(all_fpi_epoch)
    all_fpi_epoch = all_fpi_epoch[sort_idx]
    all_fpi_flux = all_fpi_flux[sort_idx]

    # Clean fill values and non-physical values
    all_fpi_flux[all_fpi_flux <= 0] = np.nan
    all_fpi_flux[all_fpi_flux >= 1e25] = np.nan

    # -------------------------------------------------------------------------
    # 2. Parse MEC Ephemeris (Position -> R [R_E], Local Time)
    # -------------------------------------------------------------------------
    coord = coord.lower()
    coord_label = 'GSE LT' if coord == 'gse' else 'MLT'

    mec_epochs_list = []
    mec_r_list = []
    mec_lt_list = []

    print(f"[Parser] Reading {len(mec_files)} MEC CDF files (Coordinate: {coord.upper()})...")
    for f in mec_files:
        try:
            cdf = cdflib.CDF(f)
            zvars = set(cdf.cdf_info().zVariables)

            ep_var = next((v for v in ['Epoch', 'epoch'] if v in zvars), None)
            if not ep_var:
                continue
            raw_ep = cdf.varget(ep_var)
            if raw_ep is None or np.size(raw_ep) == 0:
                continue

            u_sec = np.atleast_1d(cdflib.cdfepoch.unixtime(raw_ep))
            if u_sec.size == 0:
                continue

            if coord == 'gse':
                r_var = next((v for v in [f'mms{probe}_mec_r_gse', 'mms1_mec_r_gse',
                                          f'mms{probe}_mec_r_gsm', 'mms1_mec_r_gsm'] if v in zvars), None)
            else:
                r_var = next((v for v in [f'mms{probe}_mec_r_gsm', 'mms1_mec_r_gsm',
                                          f'mms{probe}_mec_r_gse', 'mms1_mec_r_gse'] if v in zvars), None)

            if not r_var:
                continue

            r_raw = cdf.varget(r_var)
            if r_raw is None or np.size(r_raw) == 0:
                continue
            r_vec = np.asarray(r_raw, dtype=np.float64) # (N, 3) in km
            if r_vec.ndim == 1:
                r_vec = r_vec.reshape(1, -1)
            if r_vec.shape[1] != 3:
                continue

            bad_r = np.abs(r_vec[:, 0]) > 1e10
            r_vec[bad_r] = np.nan
            r_re = np.atleast_1d(np.linalg.norm(r_vec, axis=1) / 6371.2) # Earth radius in km

            if coord == 'gse':
                # GSE Local Time from X_gse, Y_gse in hours [0, 24)
                phi_gse = np.arctan2(r_vec[:, 1], r_vec[:, 0])
                lt_val = np.atleast_1d((phi_gse * 12.0 / np.pi + 12.0) % 24.0)
            else:
                mlt_var = next((v for v in [f'mms{probe}_mec_mlt', 'mms1_mec_mlt'] if v in zvars), None)
                if mlt_var:
                    raw_mlt = cdf.varget(mlt_var)
                    if raw_mlt is not None and np.size(raw_mlt) > 0:
                        lt_val = np.atleast_1d(np.asarray(raw_mlt, dtype=np.float64))
                        bad_m = (lt_val < 0) | (lt_val > 24)
                        lt_val[bad_m] = np.nan
                    else:
                        phi_gsm = np.arctan2(r_vec[:, 1], r_vec[:, 0])
                        lt_val = np.atleast_1d((phi_gsm * 12.0 / np.pi + 12.0) % 24.0)
                else:
                    phi_gsm = np.arctan2(r_vec[:, 1], r_vec[:, 0])
                    lt_val = np.atleast_1d((phi_gsm * 12.0 / np.pi + 12.0) % 24.0)

            if len(u_sec) != len(r_re) or len(u_sec) != len(lt_val):
                print(f"[Parser] Warning: Skipping {os.path.basename(f)} due to MEC array length mismatch.")
                continue

            mec_epochs_list.append(u_sec)
            mec_r_list.append(r_re)
            mec_lt_list.append(lt_val)
        except Exception as e:
            print(f"[Parser] Warning: Skipping corrupted or unreadable MEC file {f}: {e}")

    if not mec_epochs_list:
        raise ValueError("Could not extract any valid MEC orbit data from provided CDF files.")

    all_mec_epoch = np.concatenate(mec_epochs_list)
    all_mec_r = np.concatenate(mec_r_list)
    all_mec_lt = np.concatenate(mec_lt_list)

    # Sort MEC timestamps
    m_sort = np.argsort(all_mec_epoch)
    all_mec_epoch = all_mec_epoch[m_sort]
    all_mec_r = all_mec_r[m_sort]
    all_mec_lt = all_mec_lt[m_sort]

    # Filter any NaNs in MEC prior to interpolation
    valid_mec = np.isfinite(all_mec_epoch) & np.isfinite(all_mec_r) & np.isfinite(all_mec_lt)
    mec_ep_clean = all_mec_epoch[valid_mec]
    mec_r_clean = all_mec_r[valid_mec]
    mec_lt_clean = all_mec_lt[valid_mec]

    # -------------------------------------------------------------------------
    # 3. Interpolate Orbit Parameters onto FPI Timestamps
    # -------------------------------------------------------------------------
    print(f"[Parser] Interpolating R [R_E] and {coord_label} onto {len(all_fpi_epoch)} FPI timestamps...")
    f_r = interp1d(mec_ep_clean, mec_r_clean, bounds_error=False, fill_value='extrapolate')
    fpi_r = f_r(all_fpi_epoch)

    # Angular unwrapping for seamless interpolation across midnight / 24h -> 0h
    lt_rad = np.unwrap(mec_lt_clean * (2.0 * np.pi / 24.0))
    f_lt = interp1d(mec_ep_clean, lt_rad, bounds_error=False, fill_value='extrapolate')
    fpi_lt = (f_lt(all_fpi_epoch) * 24.0 / (2.0 * np.pi)) % 24.0

    # Convert timestamps to matplotlib date numbers
    dt_objects = np.array([datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in all_fpi_epoch])
    times_num = mdates.date2num(dt_objects)

    # -------------------------------------------------------------------------
    # 4. Optional Time Clipping (Both FPI and MEC)
    # -------------------------------------------------------------------------
    if clip_start_dt is not None or clip_end_dt is not None:
        # FPI clipping
        fpi_mask = np.ones(len(dt_objects), dtype=bool)
        if clip_start_dt is not None:
            c_start = clip_start_dt.replace(tzinfo=datetime.timezone.utc) if clip_start_dt.tzinfo is None else clip_start_dt
            fpi_mask &= (dt_objects >= c_start)
        if clip_end_dt is not None:
            c_end = clip_end_dt.replace(tzinfo=datetime.timezone.utc) if clip_end_dt.tzinfo is None else clip_end_dt
            fpi_mask &= (dt_objects <= c_end)

        dt_objects = dt_objects[fpi_mask]
        times_num = times_num[fpi_mask]
        all_fpi_epoch = all_fpi_epoch[fpi_mask]
        all_fpi_flux = all_fpi_flux[fpi_mask]
        fpi_r = fpi_r[fpi_mask]
        fpi_lt = fpi_lt[fpi_mask]

        # MEC clipping
        mec_mask = np.ones(len(mec_ep_clean), dtype=bool)
        if clip_start_dt is not None:
            c_start_ts = clip_start_dt.replace(tzinfo=datetime.timezone.utc).timestamp()
            mec_mask &= (mec_ep_clean >= c_start_ts)
        if clip_end_dt is not None:
            c_end_ts = clip_end_dt.replace(tzinfo=datetime.timezone.utc).timestamp()
            mec_mask &= (mec_ep_clean <= c_end_ts)
        mec_ep_plot = mec_ep_clean[mec_mask]
        mec_r_plot = mec_r_clean[mec_mask]
        mec_lt_plot = mec_lt_clean[mec_mask]
    else:
        mec_ep_plot = mec_ep_clean
        mec_r_plot = mec_r_clean
        mec_lt_plot = mec_lt_clean

    # Convert MEC timestamps to matplotlib date numbers for ephemeris interpolation
    mec_dt = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in mec_ep_plot]
    mec_times_num = mdates.date2num(np.array(mec_dt)) if len(mec_dt) > 0 else np.array([])

    print(f"[Parser] Successfully loaded {len(dt_objects)} records covering "
          f"{dt_objects[0].strftime('%Y-%m-%d %H:%M:%S')} to {dt_objects[-1].strftime('%Y-%m-%d %H:%M:%S')} UTC.")

    return {
        'times_dt': dt_objects,
        'times_num': times_num,
        'raw_epoch': all_fpi_epoch,
        'flux': all_fpi_flux,            # Shape: (N_time, 32)
        'energy': fpi_energies,          # Shape: (32,) in eV
        'r_re': fpi_r,                   # Radial distance in R_E at FPI times
        'lt': fpi_lt,                    # Local Time (GSE LT or GSM MLT) in hours [0, 24)
        'mlt': fpi_lt,                   # Backward compatibility alias
        'coord': coord,                  # 'gse' or 'gsm'
        'coord_label': coord_label,      # 'GSE LT' or 'MLT'
        'mec_epochs': mec_ep_plot,       # Full MEC timestamps
        'mec_times_num': mec_times_num,  # Full MEC matplotlib date numbers
        'mec_r': mec_r_plot,             # Full MEC radial distance
        'mec_lt': mec_lt_plot,           # Full MEC Local Time (GSE LT or GSM MLT)
        'mec_mlt': mec_lt_plot,          # Backward compatibility alias
        'probe': probe
    }


def extract_flux_at_energy(energy_eV, flux_2d, target_energy_eV):
    """
    Extracts an ion flux slice at target_energy_eV using log-log interpolation across energy bins.
    """
    n_times = flux_2d.shape[0]
    flux_target = np.full(n_times, np.nan)

    log_e_grid = np.log10(energy_eV)
    log_target_e = np.log10(target_energy_eV)

    for i in range(n_times):
        f_row = flux_2d[i, :]
        valid = np.isfinite(f_row) & (f_row > 0)
        if np.sum(valid) >= 2:
            f_log_interp = interp1d(
                log_e_grid[valid],
                np.log10(f_row[valid]),
                bounds_error=False,
                fill_value=np.nan
            )
            flux_target[i] = 10.0 ** f_log_interp(log_target_e)

    return flux_target


def plot_mms1_fpi_and_orbit(data_dict,
                            target_energy_keV=1.0,
                            requested_start_dt=None,
                            requested_end_dt=None,
                            noon_loc='top',
                            point_size=None,
                            alpha=1.0,
                            step=None,
                            max_orbit_points=None,
                            output_path='mms1_fpi_ion_orbit.png',
                            dpi=300):
    """
    Generates a publication-quality 2-panel figure:
      Panel 1: MMS FPI Fast Survey Ion Differential Energy Flux Spectrogram with
               adaptive multi-row X-axis ticks (Date, UTC, R [R_E], MLT).
      Panel 2: Polar Orbit Plot (Radial Distance R vs MLT) colored by ion flux at target energy.
               Adaptive radial scaling automatically supports both Phase 1 (~12 Re) and Phase 2 (~25 Re).
    """
    times_num = data_dict['times_num']
    flux = data_dict['flux']
    energy_eV = data_dict['energy']
    r_re = data_dict['r_re']
    lt = data_dict.get('lt', data_dict.get('mlt'))
    coord = data_dict.get('coord', 'gse').lower()
    coord_label = data_dict.get('coord_label', 'GSE LT' if coord == 'gse' else 'MLT')
    probe = data_dict.get('probe', '1')

    target_energy_eV = target_energy_keV * 1000.0

    # Determine X-axis time window from user request or data span
    if requested_start_dt is not None:
        x_min = mdates.date2num(requested_start_dt)
    else:
        x_min = times_num[0]

    if requested_end_dt is not None:
        x_max = mdates.date2num(requested_end_dt)
    else:
        x_max = times_num[-1]

    total_hours = (x_max - x_min) * 24.0

    if requested_start_dt and requested_end_dt:
        if requested_start_dt.strftime('%Y-%m-%d') == requested_end_dt.strftime('%Y-%m-%d'):
            date_title = requested_start_dt.strftime('%Y-%m-%d')
        else:
            date_title = f"{requested_start_dt.strftime('%Y-%m-%d')} to {requested_end_dt.strftime('%Y-%m-%d')}"
    else:
        t_start_dt = mdates.num2date(times_num[0])
        t_end_dt = mdates.num2date(times_num[-1])
        if t_start_dt.strftime('%Y-%m-%d') == t_end_dt.strftime('%Y-%m-%d'):
            date_title = t_start_dt.strftime('%Y-%m-%d')
        else:
            date_title = f"{t_start_dt.strftime('%Y-%m-%d')} to {t_end_dt.strftime('%Y-%m-%d')}"

    # -------------------------------------------------------------------------
    # Downsample Orbit Points for High-Performance Rendering (듬성듬성 그리기)
    # -------------------------------------------------------------------------
    n_total_pts = len(lt)
    n_days = max(1.0, total_hours / 24.0)
    if max_orbit_points is not None:
        effective_max = max_orbit_points
    else:
        # Dynamically scale ~100 points/day (~100 pts/orbit), clamped between 3,000 and 45,000
        effective_max = int(np.clip(n_days * 100, 3000, 45000))

    if step is not None and step > 1:
        stride = int(step)
    elif n_total_pts > effective_max:
        stride = int(np.ceil(n_total_pts / effective_max))
    else:
        stride = 1

    if stride > 1:
        print(f"[Plotting] Fast mode: Downsampling polar orbit points by factor {stride} ({n_total_pts} -> {len(lt[::stride])} points)...")
        lt_orbit = lt[::stride]
        r_orbit = r_re[::stride]
        flux_orbit = flux[::stride]
    else:
        lt_orbit = lt
        r_orbit = r_re
        flux_orbit = flux

    print(f"[Plotting] Extracting ion flux slice at ~{target_energy_keV:.2f} keV ({len(lt_orbit)} points)...")
    flux_slice = extract_flux_at_energy(energy_eV, flux_orbit, target_energy_eV)

    # -------------------------------------------------------------------------
    # Uniform Time-Grid Resampling for Spectrogram (Handling Data Gaps cleanly)
    # Adaptive grid resolution bounds number of bins for multi-month spans
    # -------------------------------------------------------------------------
    grid_dt_sec = max(60.0, (total_hours * 3600.0) / 4000.0)
    grid_dt_num = grid_dt_sec / 86400.0
    t_grid_num = np.arange(x_min, x_max + grid_dt_num, grid_dt_num)
    n_grid_times = len(t_grid_num)

    e_grid_keV = energy_eV / 1000.0
    flux_grid = np.full((n_grid_times, len(energy_eV)), np.nan)

    # Bin raw measurements onto uniform time grid
    bin_indices = np.digitize(times_num, t_grid_num) - 1
    for k in range(len(energy_eV)):
        col_flux = flux[:, k]
        valid = np.isfinite(col_flux) & (col_flux > 0)
        if np.any(valid):
            b_valid = bin_indices[valid]
            f_valid = col_flux[valid]
            inside = (b_valid >= 0) & (b_valid < n_grid_times)
            if np.any(inside):
                counts = np.bincount(b_valid[inside], minlength=n_grid_times)
                sums = np.bincount(b_valid[inside], weights=f_valid[inside], minlength=n_grid_times)
                with np.errstate(divide='ignore', invalid='ignore'):
                    means = sums / counts
                    means[counts == 0] = np.nan
                flux_grid[:, k] = means

    # Color scaling limits (5th to 98th percentile of positive finite flux)
    valid_flux = flux_grid[np.isfinite(flux_grid) & (flux_grid > 0)]
    if len(valid_flux) > 0:
        cmin = max(1e3, 10.0 ** np.floor(np.log10(np.percentile(valid_flux, 5))))
        cmax = min(1e9, 10.0 ** np.ceil(np.log10(np.percentile(valid_flux, 98))))
    else:
        cmin, cmax = 1e4, 1e8

    norm = mcolors.LogNorm(vmin=cmin, vmax=cmax)
    cmap = plt.cm.turbo

    # Create figure
    fig = plt.figure(figsize=(11.5, 11.5), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.48)

    # -------------------------------------------------------------------------
    # Panel 1: Energy vs Time Spectrogram
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    pcm = ax1.pcolormesh(t_grid_num, e_grid_keV, flux_grid.T, shading='auto', norm=norm, cmap=cmap, rasterized=True)
    ax1.set_yscale('log')
    ax1.set_ylim(e_grid_keV[0], e_grid_keV[-1])
    ax1.set_ylabel('Ion Energy [keV]', fontsize=11, fontweight='bold')
    ax1.set_title(f'MMS{probe} FPI Fast Survey Ion Differential Energy Flux & Orbit ({date_title})',
                  fontsize=12, fontweight='bold', pad=12)

    cbar1 = fig.colorbar(pcm, ax=ax1, pad=0.02, aspect=20)
    cbar1.set_label('DEF [keV/(cm² s sr keV)]', fontsize=10, fontweight='bold')

    ax1.set_xlim(x_min, x_max)

    # Adaptive X-axis tick intervals based on total duration
    if total_hours <= 12:
        locator = mdates.HourLocator(interval=2)
        is_multi_day = False
    elif total_hours <= 24:
        locator = mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21])
        is_multi_day = False
    elif total_hours <= 48:
        locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
        is_multi_day = True
    elif total_hours <= 96:
        locator = mdates.HourLocator(interval=12)
        is_multi_day = True
    elif total_hours <= 24 * 14:
        locator = mdates.DayLocator(interval=1)
        is_multi_day = True
    elif total_hours <= 24 * 45:
        locator = mdates.DayLocator(interval=5)
        is_multi_day = True
    elif total_hours <= 24 * 120:
        locator = mdates.DayLocator(interval=15)
        is_multi_day = True
    else:
        locator = mdates.MonthLocator(interval=1)
        is_multi_day = True

    ax1.xaxis.set_major_locator(locator)
    fig.canvas.draw()
    ticks = ax1.get_xticks()
    tick_labels = []
    lt_unwrapped = np.unwrap(lt * (2.0 * np.pi / 24.0))

    has_full_mec = ('mec_lt' in data_dict and 'mec_r' in data_dict and len(data_dict['mec_lt']) > 0)
    if not has_full_mec and 'mec_mlt' in data_dict:
        has_full_mec = ('mec_mlt' in data_dict and 'mec_r' in data_dict and len(data_dict['mec_mlt']) > 0)

    mec_times_num = data_dict.get('mec_times_num', np.array([]))
    if has_full_mec:
        m_lt = data_dict.get('mec_lt', data_dict.get('mec_mlt'))
        m_r = data_dict['mec_r']
        lt_unwrapped_mec = np.unwrap(m_lt * (2.0 * np.pi / 24.0))

    for t in ticks:
        if (x_min - 1e-5) <= t <= (x_max + 1e-4):
            # Prioritize full MEC ephemeris so R and LT are shown even during survey gaps
            if has_full_mec and len(mec_times_num) > 1 and mec_times_num[0] <= t <= mec_times_num[-1]:
                r_val = float(np.interp(t, mec_times_num, m_r))
                lt_val = float((np.interp(t, mec_times_num, lt_unwrapped_mec) * 24.0 / (2.0 * np.pi)) % 24.0)
            else:
                t_eval = min(max(t, times_num[0]), times_num[-1])
                r_val = float(np.interp(t_eval, times_num, r_re))
                lt_val = float((np.interp(t_eval, times_num, lt_unwrapped) * 24.0 / (2.0 * np.pi)) % 24.0)

            dt_obj = mdates.num2date(t)
            if is_multi_day:
                if total_hours > 24 * 60:
                    time_str = dt_obj.strftime('%Y-%m-%d')
                else:
                    time_str = dt_obj.strftime('%m-%d\n%H:%M')
            else:
                time_str = dt_obj.strftime('%H:%M')
            tick_labels.append(f'{time_str}\n{r_val:.1f}\n{lt_val:.1f}')
        else:
            tick_labels.append('')

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels, fontsize=8.5)

    if is_multi_day:
        ax1.text(-0.016, -0.05, f'Date:\nUTC:\nR [RE]:\n{coord_label}:', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=8.5, fontweight='bold', linespacing=1.35)
    else:
        ax1.text(-0.016, -0.05, f'UTC:\nR [RE]:\n{coord_label}:', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=8.5, fontweight='bold', linespacing=1.35)

    ax1.grid(True, which='both', color='gray', linestyle=':', alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 2: Radial Distance R vs. Local Time Polar Plot
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], projection='polar')

    noon_loc = noon_loc.lower()
    if coord == 'gse':
        if noon_loc == 'bottom':
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(1) # CCW: 06 LT at Left, 18 LT at Right
            hour_labels = ['00 LT\n(-X GSE / Tail)', '03', '06 LT\n(-Y GSE / Dawn)', '09',
                           '12 LT\n(+X GSE / Sun)', '15', '18 LT\n(+Y GSE / Dusk)', '21']
        else:
            # Standard View: 12 LT (+X Sun) at Top, 00 LT (-X Tail) at Bottom
            ax2.set_theta_zero_location('S')
            ax2.set_theta_direction(1) # CCW: 06 LT (Dawn) at Right, 18 LT (Dusk) at Left
            hour_labels = ['00 LT\n(-X GSE / Tail)', '03', '06 LT\n(-Y GSE / Dawn)', '09',
                           '12 LT\n(+X GSE / Sun)', '15', '18 LT\n(+Y GSE / Dusk)', '21']
    else:
        if noon_loc == 'bottom':
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(1) # CCW: 06 MLT at Left, 18 MLT at Right
            hour_labels = ['00 MLT\n(Midnight)', '03', '06 MLT\n(Dawn)', '09',
                           '12 MLT\n(Noon)', '15', '18 MLT\n(Dusk)', '21']
        else:
            # Standard Magnetospheric View: 12 MLT (Noon) at North/Top, 00 MLT at South/Bottom
            ax2.set_theta_zero_location('S')
            ax2.set_theta_direction(1) # CCW: 06 MLT (Dawn) at Right, 18 MLT (Dusk) at Left
            hour_labels = ['00 MLT\n(Midnight)', '03', '06 MLT\n(Dawn)', '09',
                           '12 MLT\n(Noon)', '15', '18 MLT\n(Dusk)', '21']

    # Angular coordinate in radians from local time [0, 24)
    theta = lt_orbit * (2.0 * np.pi / 24.0)

    # Continuous trajectory underlay (using full MEC ephemeris if available)
    if has_full_mec:
        theta_mec = np.unwrap(m_lt * (2.0 * np.pi / 24.0))
        mec_stride = max(1, int(np.ceil(len(m_r) / 4000)))
        ax2.plot(theta_mec[::mec_stride], m_r[::mec_stride], color='#777777', lw=0.6, alpha=0.35, zorder=1, label='Full Orbit Track')
    else:
        ax2.plot(theta, r_orbit, color='gray', lw=0.45, alpha=0.3, zorder=1, label='Orbit Track')

    # Scatter points colored by ion flux at chosen energy
    n_pts = len(theta)
    if point_size is not None:
        pt_size = point_size
    else:
        if n_pts <= 3000:
            pt_size = 3.5
        elif n_pts <= 10000:
            pt_size = 2.0
        elif n_pts <= 25000:
            pt_size = 1.2
        else:
            pt_size = 0.6

    sc = ax2.scatter(theta, r_orbit, c=flux_slice, cmap=cmap, norm=norm,
                     s=pt_size, alpha=alpha, edgecolors='none', zorder=3,
                     rasterized=True,
                     label=f'Flux @ ~{target_energy_keV:.2f} keV')

    # Dynamic radial limit scaling supporting both Phase 1 (~12 Re) and Phase 2 (~25 Re)
    r_candidates = [float(np.nanmax(r_orbit))]
    if has_full_mec and len(m_r) > 0:
        r_candidates.append(float(np.nanmax(m_r)))
    peak_r = max(r_candidates)

    if peak_r <= 13.5:
        # Phase 1 apogee
        r_lim = 13.0
        r_ticks = [2, 4, 6, 8, 10, 12]
    elif peak_r <= 18.0:
        r_lim = float(np.ceil(peak_r / 2.0) * 2 + 1.0)
        r_ticks = [4, 8, 12, 16]
    else:
        # Phase 2 apogee (up to ~25 Re)
        r_lim = float(np.ceil((peak_r + 1.0) / 5.0) * 5.0)
        r_ticks = list(range(5, int(r_lim) + 1, 5))

    ax2.set_ylim(0, r_lim)
    ax2.set_yticks(r_ticks)
    ax2.set_yticklabels([f'{r} RE' for r in r_ticks], fontsize=8, color='#444444', fontweight='semibold')
    ax2.set_rlabel_position(315)

    # Azimuthal ticks (8 sectors: every 3 hours)
    az_ticks = np.linspace(0, 2.0 * np.pi, 8, endpoint=False)
    ax2.set_xticks(az_ticks)
    ax2.set_xticklabels(hour_labels, fontsize=9, fontweight='bold')

    # Draw Earth disk at center (R <= 1.0 Re)
    theta_earth = np.linspace(0, 2.0 * np.pi, 200)
    ax2.fill(theta_earth, np.ones_like(theta_earth) * 1.0, color='#87CEEB', alpha=0.45, zorder=2)
    ax2.plot(theta_earth, np.ones_like(theta_earth) * 1.0, color='#1E90FF', lw=1.2, zorder=2)
    ax2.text(0, 0, 'Earth\n(1 RE)', ha='center', va='center', fontsize=7, fontweight='bold', color='#104E8B', zorder=4)

    # Mark Start (green square) and End (red triangle) positions of the orbit
    if has_full_mec:
        ax2.scatter(theta_mec[0], m_r[0], marker='s', s=45, color='lime', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Orbit Start')
        ax2.scatter(theta_mec[-1], m_r[-1], marker='^', s=55, color='red', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Orbit End')
    else:
        ax2.scatter(theta[0], r_orbit[0], marker='s', s=45, color='lime', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Start')
        ax2.scatter(theta[-1], r_orbit[-1], marker='^', s=55, color='red', edgecolors='black',
                    linewidths=1.2, zorder=5, label='End')

    ax2.grid(True, color='gray', linestyle=':', alpha=0.6)
    if coord == 'gse':
        title_text = f'MMS{probe} Polar Orbit in GSE: R [RE] vs. GSE Local Time (Colored by Ion Flux at {target_energy_keV:.2f} keV)'
    else:
        title_text = f'MMS{probe} Polar Orbit in GSM: R [RE] vs. MLT (Colored by Ion Flux at {target_energy_keV:.2f} keV)'
    ax2.set_title(title_text, fontsize=11.5, fontweight='bold', pad=22)

    # Polar plot colorbar
    cbar2 = fig.colorbar(sc, ax=ax2, pad=0.08, aspect=25, shrink=0.75)
    cbar2.set_label(f'DEF @ ~{target_energy_keV:.2f} keV [keV/(cm² s sr keV)]', fontsize=9.5, fontweight='bold')

    # Legend in lower right corner
    ax2.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=8.5, framealpha=0.9)

    # Resilient saving with fallback for Windows/OneDrive locks
    print(f"[Plotting] Saving figure to {output_path} (DPI={dpi})...")
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    except OSError as err:
        print(f"[Plotting] Warning: Direct write to {output_path} hit file lock ({err}). Using atomic temp write...")
        import tempfile, shutil, time
        tmp_fd, tmp_file = tempfile.mkstemp(suffix='.png')
        os.close(tmp_fd)
        plt.savefig(tmp_file, dpi=dpi, bbox_inches='tight')
        time.sleep(0.5)
        try:
            shutil.move(tmp_file, output_path)
        except Exception:
            alt_path = output_path.replace('.png', f'_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            shutil.move(tmp_file, alt_path)
            output_path = alt_path
            print(f"[Plotting] Saved to alternative file: {output_path}")

    plt.close(fig)
    print(f"[Plotting] Successfully created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MMS1 FPI Ion Spectrogram and Orbital Trajectory."
    )
    parser.add_argument('--probe', type=str, default='1',
                        help="MMS Spacecraft number (1, 2, 3, 4). Default: 1")
    parser.add_argument('--date', nargs='*', default=['2015-12-30', '2015-12-31'],
                        help="Single date or multi-day range (e.g. '2015-12-30 2015-12-31'). Default: 2015-12-30 2015-12-31")
    parser.add_argument('--trange', nargs='*', default=None,
                        help="Time range strings (e.g. '2015-12-30 2016-01-01')")
    parser.add_argument('--energy', type=float, default=1.0,
                        help="Target ion energy for polar orbit slice in keV (default: 1.0 keV)")
    parser.add_argument('--point-size', type=float, default=None,
                        help="Scatter point size on polar orbit plot (e.g. 0.8, 1.2, 2.0). Default: adaptive thin size.")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Scatter point opacity (alpha) on polar orbit plot (default: 1.0, fully opaque / no transparency).")
    parser.add_argument('--step', type=int, default=None,
                        help="Step/stride to downsample orbit points (e.g. --step 5 draws every 5th point). Default: auto-downsamples based on duration.")
    parser.add_argument('--max-orbit-points', type=int, default=None,
                        help="Maximum points to render on polar orbit plot (default: dynamic ~100 pts/day, 3000-45000).")
    parser.add_argument('--download', action='store_true',
                        help="Auto-download missing CDF files if not cached locally.")
    parser.add_argument('--noon', type=str, default='top', choices=['top', 'bottom'],
                        help="Orientation of Noon / Sun (12 LT): 'top' or 'bottom'")
    parser.add_argument('--coord', type=str, default='gse', choices=['gse', 'gsm', 'GSE', 'GSM'],
                        help="Coordinate system for orbit plot: 'gse' (Geocentric Solar Ecliptic, smooth) or 'gsm' (Geocentric Solar Magnetospheric / SM MLT). Default: 'gse'")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Path to directory containing downloaded MMS CDF files")
    parser.add_argument('--output', type=str, default='mms1_fpi_ion_orbit.png',
                        help="Output image filepath (default: mms1_fpi_ion_orbit.png)")
    parser.add_argument('--dpi', type=int, default=300,
                        help="Output image resolution in DPI (default: 300)")

    args = parser.parse_args()

    probe = normalize_probe_name(args.probe)
    dt_start, dt_end, calendar_dates = parse_date_or_trange(args.date, args.trange)

    pts_str = f"{args.max_orbit_points}" if args.max_orbit_points is not None else "Dynamic (~100 pts/day, 3k-45k)"
    print(f"========================================================================")
    print(f" MMS{probe} FPI Fast Survey Ion Spectrogram & Orbit Plotter")
    print(f" Target Period: {dt_start.strftime('%Y-%m-%d %H:%M:%S')} to {dt_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Calendar Days: {len(calendar_dates)} day(s) -> {calendar_dates[0]} to {calendar_dates[-1]}")
    print(f" Coordinate:    {args.coord.upper()}")
    print(f" Target Energy: {args.energy:.2f} keV")
    print(f" Polar Noon:    {args.noon}")
    print(f" Fast Render:   Max Points={pts_str} | Step={args.step if args.step else 'Auto'}")
    print(f" Output File:   {args.output}")
    print(f"========================================================================")

    # 1. Rapidly discover local CDF files (or download if explicitly enabled)
    fpi_files, mec_files = find_local_mms_files(calendar_dates, probe=probe, data_dir=args.data_dir, allow_download=args.download)

    # 2. Load and interpolate dataset
    data_dict = load_mms_fpi_and_orbit(
        fpi_files=fpi_files,
        mec_files=mec_files,
        probe=probe,
        coord=args.coord.lower(),
        clip_start_dt=dt_start,
        clip_end_dt=dt_end
    )

    # 3. Generate high-resolution publication plot
    plot_mms1_fpi_and_orbit(
        data_dict=data_dict,
        target_energy_keV=args.energy,
        requested_start_dt=dt_start,
        requested_end_dt=dt_end,
        noon_loc=args.noon,
        point_size=args.point_size,
        alpha=args.alpha,
        step=args.step,
        max_orbit_points=args.max_orbit_points,
        output_path=args.output,
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()


