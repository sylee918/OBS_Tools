#!/usr/bin/env python3
"""
plot_themis_esa_proton_orbit.py
================================
Visualizes THEMIS (Time History of Events and Macroscale Interactions during Substorms)
ESA (Electrostatic Analyzer) Ion Spectrogram alongside spacecraft orbital parameters
(Radial Distance R [R_E], GSE LT / MLT) over single-day or multi-day time intervals.

Supported Spacecraft:
  - THEMIS-A (tha, default), THEMIS-B (thb), THEMIS-C (thc), THEMIS-D (thd), THEMIS-E (the).

Key Features:
  1. Multi-Day / Time-Range Support:
     - Accepts single date (`--date 2018-01-01`) or multi-day range (`--date 2018-01-01 2018-01-03`
       or `--trange 2018-01-01 2018-01-04`).
     - Defaults to 2018-01-01 through 2018-01-03 if no date is specified.
     - Smoothly concatenates multi-day intervals into a continuous time series.
  2. Coordinate Systems (GSE default, GSM optional via `--coord`):
     - GSE (default): Geocentric Solar Ecliptic, smooth orbital trajectory free of dipole wobble.
     - GSM: Geocentric Solar Magnetospheric, SM-based MLT reflecting geomagnetic boundary physics.
  3. Top Panel: Time-Energy Spectrogram with adaptive multi-parameter X-axis:
     - Date (for multi-day spans)
     - UTC (HH:MM)
     - Radial distance R [R_E]
     - GSE LT (or MLT in GSM mode)
  4. Bottom Panel: Radial Distance R [R_E] vs. Local Time Polar Orbit Trajectory:
     - Colored by ion differential energy flux at a selected energy band (~1.0 keV default, configurable).
     - Standard View: Sun / Noon (12 LT) at Top, Tail / Midnight (00 LT) at Bottom,
       Dawn (06 LT) at Right, Dusk (18 LT) at Left (configurable via `--noon top` or `--noon bottom`).
     - Earth boundary circle at R <= 1 R_E and adaptive radial distance rings.
     - User-controllable point size (`--point-size`) for thin, razor-sharp orbit curves.
     - 15-minute fixed cadence polar orbit sampling without point count limit (`--interval 15.0`).
  5. High-Performance Direct CDF Parsing & Auto-Download:
     - Discovers local THEMIS CDF files or downloads missing files via pyspedas.
     - Parses CDFs directly via `cdflib` for maximum speed and robustness.

Usage Examples:
  python plot_themis_esa_proton_orbit.py --probe a --date 2018-01-01 2018-01-03
  python plot_themis_esa_proton_orbit.py --probe a --date 2018-01-01 --coord gse --energy 1.0
  python plot_themis_esa_proton_orbit.py --probe a --trange 2018-01-01 2018-01-04 --point-size 1.0
"""

import os
import sys
import glob
import argparse
import datetime
import tempfile
import shutil
import time

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import cdflib


# =============================================================================
# 1. Date & Argument Parsing
# =============================================================================

def parse_date_arguments(raw_args):
    """
    Parses single date, multi-day range, or ISO timestamps into:
      dt_start: datetime.datetime
      dt_end: datetime.datetime
      calendar_dates: list of 'YYYY-MM-DD' strings covering every calendar day.
    """
    if not raw_args:
        raw_args = ['2018-01-01', '2018-01-03']

    date_formats = [
        '%Y-%m-%d',
        '%Y-%m-%d/%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y%m%d'
    ]

    def parse_single_str(s):
        s = s.strip()
        for fmt in date_formats:
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


# =============================================================================
# 2. File Discovery & Automatic Download
# =============================================================================

def build_themis_file_index(search_dirs, probe='a'):
    """
    Performs a high-speed, single-pass directory scan over all search directories.
    Indexes files by date key (YYYYMMDD) in < 1 second.
    """
    esa_index = {}
    state_index = {}
    p = probe.lower()
    esa_prefix = f'th{p}_l2_esa_'
    state_prefix = f'th{p}_l1_state_'

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

                if fname.startswith(esa_prefix):
                    parts = fname[len(esa_prefix):].split('_')
                    if parts and len(parts[0]) >= 8:
                        date_key = parts[0][:8]
                        esa_index.setdefault(date_key, []).append(fpath)

                elif fname.startswith(state_prefix):
                    parts = fname[len(state_prefix):].split('_')
                    if parts and len(parts[0]) >= 8:
                        date_key = parts[0][:8]
                        state_index.setdefault(date_key, []).append(fpath)

    for k in esa_index:
        esa_index[k].sort()
    for k in state_index:
        state_index[k].sort()

    return esa_index, state_index


def find_local_themis_files(calendar_dates, probe='a', data_dir=None, allow_download=False):
    """
    Rapidly locates local THEMIS State and ESA CDF files using 1-pass directory indexing (< 1s).
    If allow_download is False and files are missing, reports missing dates immediately
    without wasting time querying remote servers.
    """
    probe = probe.lower()
    search_dirs = []
    if data_dir:
        search_dirs.append(os.path.normpath(data_dir))
    search_dirs.extend([
        os.path.normpath(f'themis_data/th{probe}'),
        os.path.normpath('themis_data'),
        os.path.normpath(f'pydata/themis/th{probe}'),
        os.path.normpath('pydata/themis')
    ])

    print(f"[File Discovery] Rapidly indexing local directories (1-pass scan)...")
    t0 = datetime.datetime.now()
    esa_index, state_index = build_themis_file_index(search_dirs, probe=probe)
    dt_scan = (datetime.datetime.now() - t0).total_seconds()
    print(f"[File Discovery] Indexed in {dt_scan:.2f}s! Found {sum(len(v) for v in esa_index.values())} ESA files, {sum(len(v) for v in state_index.values())} State files.")

    esa_files = []
    state_files = []
    missing_dates_esa = []
    missing_dates_state = []

    for d_str in calendar_dates:
        date_compact = d_str.replace('-', '')

        if date_compact in esa_index and len(esa_index[date_compact]) > 0:
            esa_files.extend(esa_index[date_compact])
        else:
            missing_dates_esa.append(d_str)

        if date_compact in state_index and len(state_index[date_compact]) > 0:
            state_files.extend(state_index[date_compact])
        else:
            missing_dates_state.append(d_str)

    esa_files = sorted(list(set(esa_files)))
    state_files = sorted(list(set(state_files)))

    # If missing files detected
    if missing_dates_esa or missing_dates_state:
        if allow_download:
            print(f"[Downloader] --download specified: downloading missing THEMIS-{probe.upper()} data...")
            try:
                import pyspedas.projects.themis as thm
                all_missing = sorted(list(set(missing_dates_esa + missing_dates_state)))
                t_start_dl = all_missing[0]
                t_end_dl = (datetime.datetime.strptime(all_missing[-1], '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                trange_dl = [t_start_dl, t_end_dl]

                if missing_dates_state:
                    thm.state(probe=probe, trange=trange_dl, downloadonly=True)
                if missing_dates_esa:
                    thm.esa(probe=probe, trange=trange_dl, level='l2', downloadonly=True)

                # Re-index
                esa_index, state_index = build_themis_file_index(search_dirs, probe=probe)
                esa_files = []
                state_files = []
                for d_str in calendar_dates:
                    dc = d_str.replace('-', '')
                    if dc in esa_index:
                        esa_files.extend(esa_index[dc])
                    if dc in state_index:
                        state_files.extend(state_index[dc])
                esa_files = sorted(list(set(esa_files)))
                state_files = sorted(list(set(state_files)))
            except Exception as e:
                print(f"[Downloader] Download encountered an issue: {e}")
        else:
            print("\n" + "!" * 78)
            print(f"[Data Check Warning] Missing local CDF files for requested date range!")
            if missing_dates_esa:
                n_miss = len(missing_dates_esa)
                sample = missing_dates_esa[:5]
                print(f"  - Missing ESA data ({n_miss} day{'s' if n_miss>1 else ''}): {sample}{' ...' if n_miss > 5 else ''}")
            if missing_dates_state:
                n_miss = len(missing_dates_state)
                sample = missing_dates_state[:5]
                print(f"  - Missing State data ({n_miss} day{'s' if n_miss>1 else ''}): {sample}{' ...' if n_miss > 5 else ''}")
            print("\n[Recommendation] To download data fast and safely, run:")
            print(f"  python download_themis_data.py --probe {probe} --date {calendar_dates[0]} {calendar_dates[-1]}")
            print("Or add '--download' flag to this script to auto-download right now.")
            print("!" * 78 + "\n")

    if not esa_files:
        raise FileNotFoundError(f"No THEMIS-{probe.upper()} ESA CDF files found locally for dates: {calendar_dates[:3]}...")
    if not state_files:
        raise FileNotFoundError(f"No THEMIS-{probe.upper()} State CDF files found locally for dates: {calendar_dates[:3]}...")

    print(f"[File Discovery] Ready! Selected {len(esa_files)} ESA files and {len(state_files)} State files.")
    return esa_files, state_files


# =============================================================================
# 3. Direct CDF Parser & Orbit Interpolation
# =============================================================================

def load_themis_esa_and_orbit(esa_files, state_files, probe='a', coord='gse',
                              mode='reduced', clip_start_dt=None, clip_end_dt=None):
    """
    Loads THEMIS ESA ion differential energy flux and State orbital ephemeris.
    Interpolates R [R_E] and Local Time (GSE LT or GSM MLT) onto ESA timestamps.

    Parameters:
      mode: 'reduced' (peir, default continuous spin survey) or 'full' (peif).
    """
    probe = probe.lower()
    coord = coord.lower()
    coord_label = 'GSE LT' if coord == 'gse' else 'MLT'

    # -------------------------------------------------------------------------
    # 1. Parse ESA Ion Flux and Energy Data
    # -------------------------------------------------------------------------
    esa_epochs_list = []
    esa_flux_list = []
    esa_yaxis_list = []

    prefix = f'th{probe}_'
    mode_tag = 'peir' if mode.lower() == 'reduced' else 'peif'
    time_var_candidates = [f'{prefix}{mode_tag}_time', f'{prefix}peir_time', f'{prefix}peif_time']
    flux_var_candidates = [f'{prefix}{mode_tag}_en_eflux', f'{prefix}peir_en_eflux', f'{prefix}peif_en_eflux']
    yaxis_var_candidates = [f'{prefix}{mode_tag}_en_eflux_yaxis', f'{prefix}peir_en_eflux_yaxis', f'{prefix}peif_en_eflux_yaxis']

    print(f"[Parser] Reading {len(esa_files)} ESA CDF files (Probe: {probe.upper()}, Mode: {mode})...")
    for f in esa_files:
        try:
            cdf = cdflib.CDF(f)
            zvars = set(cdf.cdf_info().zVariables)

            # Find time variable
            t_var = next((v for v in time_var_candidates if v in zvars), None)
            if not t_var:
                continue
            raw_t = cdf.varget(t_var)
            if raw_t is None or np.size(raw_t) == 0:
                continue
            u_sec = np.atleast_1d(np.asarray(raw_t, dtype=np.float64))
            if u_sec.size == 0:
                continue

            # Find flux variable
            fl_var = next((v for v in flux_var_candidates if v in zvars), None)
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

            # Find yaxis energy variable
            y_var = next((v for v in yaxis_var_candidates if v in zvars), None)
            if not y_var:
                continue
            y_raw = cdf.varget(y_var)
            if y_raw is None or np.size(y_raw) == 0:
                continue
            y_arr = np.asarray(y_raw, dtype=np.float64)
            if y_arr.ndim == 1:
                # Expand static 1D energy table across all records
                y_arr = np.tile(y_arr, (fl.shape[0], 1))
            elif y_arr.ndim == 0 or y_arr.size == 0:
                continue

            if len(u_sec) != fl.shape[0] or fl.shape[0] != y_arr.shape[0]:
                print(f"[Parser] Warning: Skipping {os.path.basename(f)} due to dimension mismatch: "
                      f"Time ({len(u_sec)}), Flux ({fl.shape[0]}), YAxis ({y_arr.shape[0]}).")
                continue

            esa_epochs_list.append(u_sec)
            esa_flux_list.append(fl)
            esa_yaxis_list.append(y_arr)
        except Exception as e:
            print(f"[Parser] Warning: Skipping corrupted or unreadable ESA file {f}: {e}")

    if not esa_epochs_list:
        raise ValueError("Could not extract any valid ESA data from provided CDF files.")

    all_esa_epoch = np.concatenate(esa_epochs_list)
    all_esa_flux = np.concatenate(esa_flux_list, axis=0)
    all_esa_yaxis = np.concatenate(esa_yaxis_list, axis=0)

    # Sort chronologically
    s_idx = np.argsort(all_esa_epoch)
    all_esa_epoch = all_esa_epoch[s_idx]
    all_esa_flux = all_esa_flux[s_idx]
    all_esa_yaxis = all_esa_yaxis[s_idx]

    # Clean non-physical flux values
    all_esa_flux[all_esa_flux <= 0] = np.nan
    all_esa_flux[all_esa_flux >= 1e25] = np.nan

    # -------------------------------------------------------------------------
    # 2. Parse State Ephemeris (Position -> R [R_E], Local Time)
    # -------------------------------------------------------------------------
    state_epochs_list = []
    state_r_list = []
    state_lt_list = []

    pos_var_name = f'th{probe}_pos_gse' if coord == 'gse' else f'th{probe}_pos_gsm'
    time_var_name = f'th{probe}_state_time'

    print(f"[Parser] Reading {len(state_files)} State CDF files (Coordinate: {coord.upper()})...")
    for f in state_files:
        try:
            cdf = cdflib.CDF(f)
            zvars = set(cdf.cdf_info().zVariables)

            # Find state time
            t_var = next((v for v in [time_var_name, f'th{probe}_state_time', 'tha_state_time'] if v in zvars), None)
            if not t_var:
                continue
            raw_t = cdf.varget(t_var)
            if raw_t is None or np.size(raw_t) == 0:
                continue
            u_sec = np.atleast_1d(np.asarray(raw_t, dtype=np.float64))
            if u_sec.size == 0:
                continue

            # Find position vector (km)
            p_var = next((v for v in [pos_var_name, f'th{probe}_pos_gse', f'th{probe}_pos_gsm'] if v in zvars), None)
            if not p_var:
                continue
            raw_p = cdf.varget(p_var)
            if raw_p is None or np.size(raw_p) == 0:
                continue
            p_vec = np.asarray(raw_p, dtype=np.float64) # (N, 3) in km
            if p_vec.ndim == 1:
                p_vec = p_vec.reshape(1, -1)
            if p_vec.shape[1] != 3:
                continue

            # Bad values check
            bad_p = np.abs(p_vec[:, 0]) > 1e10
            p_vec[bad_p] = np.nan

            # Earth radius normalization (6371.2 km)
            r_re = np.atleast_1d(np.linalg.norm(p_vec, axis=1) / 6371.2)

            # Local Time calculation: phi = atan2(Y, X) -> LT = (phi * 12/pi + 12) % 24
            phi = np.arctan2(p_vec[:, 1], p_vec[:, 0])
            lt_val = np.atleast_1d((phi * 12.0 / np.pi + 12.0) % 24.0)

            if len(u_sec) != len(r_re) or len(u_sec) != len(lt_val):
                print(f"[Parser] Warning: Skipping {os.path.basename(f)} due to State array length mismatch.")
                continue

            state_epochs_list.append(u_sec)
            state_r_list.append(r_re)
            state_lt_list.append(lt_val)
        except Exception as e:
            print(f"[Parser] Warning: Skipping corrupted or unreadable State file {f}: {e}")

    if not state_epochs_list:
        raise ValueError("Could not extract any valid State orbit data from provided CDF files.")

    all_state_epoch = np.concatenate(state_epochs_list)
    all_state_r = np.concatenate(state_r_list)
    all_state_lt = np.concatenate(state_lt_list)

    s_sort = np.argsort(all_state_epoch)
    all_state_epoch = all_state_epoch[s_sort]
    all_state_r = all_state_r[s_sort]
    all_state_lt = all_state_lt[s_sort]

    # Clean NaNs in state prior to interpolation
    valid_st = np.isfinite(all_state_epoch) & np.isfinite(all_state_r) & np.isfinite(all_state_lt)
    st_ep_clean = all_state_epoch[valid_st]
    st_r_clean = all_state_r[valid_st]
    st_lt_clean = all_state_lt[valid_st]

    # -------------------------------------------------------------------------
    # 3. Interpolate Orbit Parameters onto ESA Timestamps
    # -------------------------------------------------------------------------
    print(f"[Parser] Interpolating R [R_E] and {coord_label} onto {len(all_esa_epoch)} ESA timestamps...")
    f_r = interp1d(st_ep_clean, st_r_clean, bounds_error=False, fill_value='extrapolate')
    esa_r = f_r(all_esa_epoch)

    # Angular unwrapping across midnight
    lt_rad = np.unwrap(st_lt_clean * (2.0 * np.pi / 24.0))
    f_lt = interp1d(st_ep_clean, lt_rad, bounds_error=False, fill_value='extrapolate')
    esa_lt = (f_lt(all_esa_epoch) * 24.0 / (2.0 * np.pi)) % 24.0

    # Convert timestamps to matplotlib date numbers
    dt_objects = np.array([datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in all_esa_epoch])
    times_num = mdates.date2num(dt_objects)

    # -------------------------------------------------------------------------
    # 4. Optional Time Clipping
    # -------------------------------------------------------------------------
    if clip_start_dt is not None or clip_end_dt is not None:
        esa_mask = np.ones(len(dt_objects), dtype=bool)
        if clip_start_dt is not None:
            c_start = clip_start_dt.replace(tzinfo=datetime.timezone.utc) if clip_start_dt.tzinfo is None else clip_start_dt
            esa_mask &= (dt_objects >= c_start)
        if clip_end_dt is not None:
            c_end = clip_end_dt.replace(tzinfo=datetime.timezone.utc) if clip_end_dt.tzinfo is None else clip_end_dt
            esa_mask &= (dt_objects <= c_end)

        times_num = times_num[esa_mask]
        all_esa_flux = all_esa_flux[esa_mask]
        all_esa_yaxis = all_esa_yaxis[esa_mask]
        esa_r = esa_r[esa_mask]
        esa_lt = esa_lt[esa_mask]
        all_esa_epoch = all_esa_epoch[esa_mask]

        # Clip state trajectory
        st_dt = np.array([datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in st_ep_clean])
        st_mask = np.ones(len(st_dt), dtype=bool)
        if clip_start_dt is not None:
            st_mask &= (st_dt >= c_start)
        if clip_end_dt is not None:
            st_mask &= (st_dt <= c_end)

        st_ep_plot = st_ep_clean[st_mask]
        st_r_plot = st_r_clean[st_mask]
        st_lt_plot = st_lt_clean[st_mask]
        st_times_num = mdates.date2num(st_dt[st_mask])
    else:
        st_ep_plot = st_ep_clean
        st_r_plot = st_r_clean
        st_lt_plot = st_lt_clean
        st_dt = np.array([datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in st_ep_clean])
        st_times_num = mdates.date2num(st_dt)

    if len(times_num) == 0:
        raise ValueError("No ESA data remaining after requested time range clipping.")

    return {
        'epochs': all_esa_epoch,
        'times_num': times_num,
        'flux': all_esa_flux,
        'yaxis': all_esa_yaxis, # (N, 32) energy in eV
        'r_re': esa_r,
        'lt': esa_lt,
        'coord': coord,
        'coord_label': coord_label,
        'state_epochs': st_ep_plot,
        'state_times_num': st_times_num,
        'state_r': st_r_plot,
        'state_lt': st_lt_plot,
        'probe': probe
    }


# =============================================================================
# 4. Energy Slice Extraction & Spectrogram Regridding
# =============================================================================

def extract_flux_at_energy(yaxis_eV, flux_2d, target_energy_eV):
    """
    Extracts an ion differential energy flux slice at target_energy_eV using
    log-log interpolation across energy bins for each time step.
    Handles time-varying energy tables (e.g. 24 vs 32 energy channels).
    """
    n_times = flux_2d.shape[0]
    flux_target = np.full(n_times, np.nan)
    log_target_e = np.log10(target_energy_eV)

    for i in range(n_times):
        e_row = yaxis_eV[i]
        f_row = flux_2d[i]
        valid = np.isfinite(e_row) & np.isfinite(f_row) & (f_row > 0) & (e_row > 0)
        if np.sum(valid) >= 2:
            e_val = e_row[valid]
            f_val = f_row[valid]
            s = np.argsort(e_val)
            flux_target[i] = 10.0 ** np.interp(log_target_e, np.log10(e_val[s]), np.log10(f_val[s]))

    return flux_target


def regrid_spectrogram_on_fixed_energy(yaxis_eV, flux_2d, common_energy_eV):
    """
    Interpolates each spectrum row onto a fixed master logarithmic energy grid.
    """
    n_times = flux_2d.shape[0]
    n_e = len(common_energy_eV)
    log_common_e = np.log10(common_energy_eV)
    regridded = np.full((n_times, n_e), np.nan)

    for i in range(n_times):
        e_row = yaxis_eV[i]
        f_row = flux_2d[i]
        valid = np.isfinite(e_row) & np.isfinite(f_row) & (f_row > 0) & (e_row > 0)
        if np.sum(valid) >= 2:
            e_val = e_row[valid]
            f_val = f_row[valid]
            s = np.argsort(e_val)
            regridded[i, :] = 10.0 ** np.interp(log_common_e, np.log10(e_val[s]), np.log10(f_val[s]))

    return regridded


# =============================================================================
# 5. Publication-Quality Figure Generation
# =============================================================================

def plot_themis_esa_and_orbit(data_dict,
                              target_energy_keV=1.0,
                              requested_start_dt=None,
                              requested_end_dt=None,
                              noon_loc='top',
                              point_size=None,
                              alpha=1.0,
                              interval_min=15.0,
                              step=None,
                              output_path='themis_esa_ion_orbit.png',
                              dpi=300):
    """
    Generates a 2-panel publication-grade visualization:
      Panel 1: THEMIS ESA Ion Differential Energy Flux Spectrogram with
               adaptive multi-row X-axis ticks (Date, UTC, R [R_E], Local Time).
      Panel 2: Polar Orbit Plot (R vs Local Time) colored by ion flux at chosen energy.
    """
    times_num = data_dict['times_num']
    flux = data_dict['flux']
    yaxis_eV = data_dict['yaxis']
    r_re = data_dict['r_re']
    lt = data_dict['lt']
    coord = data_dict.get('coord', 'gse').lower()
    coord_label = data_dict.get('coord_label', 'GSE LT' if coord == 'gse' else 'MLT')
    probe = data_dict.get('probe', 'a').upper()

    target_energy_eV = target_energy_keV * 1000.0

    # Determine X-axis time window
    x_min = mdates.date2num(requested_start_dt) if requested_start_dt is not None else times_num[0]
    x_max = mdates.date2num(requested_end_dt) if requested_end_dt is not None else times_num[-1]
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
    # Orbit Point Sampling (15-Minute Fixed Interval, No Point Count Limit)
    # -------------------------------------------------------------------------
    if step is not None and step > 1:
        lt_orbit = lt[::step]
        r_orbit = r_re[::step]
        yaxis_orbit = yaxis_eV[::step]
        flux_orbit = flux[::step]
        print(f"[Plotting] Custom step stride applied ({len(lt)} -> {len(lt_orbit)} points)...")
    elif interval_min is not None and interval_min > 0:
        cadence_sec = interval_min * 60.0
        t_epochs = data_dict.get('epochs', None)
        if t_epochs is not None and len(t_epochs) > 0:
            t_start_grid = np.floor(t_epochs[0] / cadence_sec) * cadence_sec
            target_grid = np.arange(t_start_grid, t_epochs[-1] + cadence_sec, cadence_sec)
            idx = np.searchsorted(t_epochs, target_grid)
            idx = np.clip(idx, 0, len(t_epochs) - 1)
            prev_idx = np.maximum(0, idx - 1)
            dist_curr = np.abs(t_epochs[idx] - target_grid)
            dist_prev = np.abs(t_epochs[prev_idx] - target_grid)
            best_idx = np.where(dist_prev < dist_curr, prev_idx, idx)

            valid = np.abs(t_epochs[best_idx] - target_grid) <= (cadence_sec / 2.0)
            sample_idx = np.unique(best_idx[valid])
            if len(sample_idx) > 0:
                lt_orbit = lt[sample_idx]
                r_orbit = r_re[sample_idx]
                yaxis_orbit = yaxis_eV[sample_idx]
                flux_orbit = flux[sample_idx]
                print(f"[Plotting] Sampled polar orbit points at {interval_min:g}-minute intervals ({len(lt)} -> {len(lt_orbit)} points, point limit removed)...")
            else:
                lt_orbit, r_orbit, yaxis_orbit, flux_orbit = lt, r_re, yaxis_eV, flux
        else:
            cadence_num = interval_min / (24.0 * 60.0)
            target_times = np.arange(times_num[0], times_num[-1] + cadence_num, cadence_num)
            idx = np.clip(np.searchsorted(times_num, target_times), 0, len(times_num) - 1)
            sample_idx = np.unique(idx)
            lt_orbit = lt[sample_idx]
            r_orbit = r_re[sample_idx]
            yaxis_orbit = yaxis_eV[sample_idx]
            flux_orbit = flux[sample_idx]
            print(f"[Plotting] Sampled polar orbit points at {interval_min:g}-minute intervals ({len(lt)} -> {len(lt_orbit)} points, point limit removed)...")
    else:
        # Full points without limitation or downsampling
        lt_orbit = lt
        r_orbit = r_re
        yaxis_orbit = yaxis_eV
        flux_orbit = flux
        print(f"[Plotting] Orbit points: Rendering all {len(lt)} points without limitation...")

    print(f"[Plotting] Extracting ion flux slice at ~{target_energy_keV:.2f} keV ({len(lt_orbit)} points)...")
    flux_slice = extract_flux_at_energy(yaxis_orbit, flux_orbit, target_energy_eV)

    # Master fixed energy grid for spectrogram (6 eV to 26 keV, 32 bins)
    common_energy_eV = np.geomspace(5.8, 26000.0, 32)
    e_grid_keV = common_energy_eV / 1000.0

    print(f"[Plotting] Regridding {len(times_num)} spectra onto master energy channels...")
    flux_regridded = regrid_spectrogram_on_fixed_energy(yaxis_eV, flux, common_energy_eV)

    # -------------------------------------------------------------------------
    # Uniform Time-Grid Resampling for Spectrogram (Handling Gaps cleanly)
    # -------------------------------------------------------------------------
    grid_dt_sec = max(30.0, (total_hours * 3600.0) / 4000.0)
    grid_dt_num = grid_dt_sec / 86400.0
    t_grid_num = np.arange(x_min, x_max + grid_dt_num, grid_dt_num)
    n_grid_times = len(t_grid_num)

    flux_grid = np.full((n_grid_times, len(common_energy_eV)), np.nan)
    bin_indices = np.digitize(times_num, t_grid_num) - 1

    for k in range(len(common_energy_eV)):
        col_flux = flux_regridded[:, k]
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

    # Color scaling limits (5th to 98th percentile)
    valid_flux = flux_grid[np.isfinite(flux_grid) & (flux_grid > 0)]
    if len(valid_flux) > 0:
        cmin = max(1e2, 10.0 ** np.floor(np.log10(np.percentile(valid_flux, 5))))
        cmax = min(1e9, 10.0 ** np.ceil(np.log10(np.percentile(valid_flux, 98))))
    else:
        cmin, cmax = 1e3, 1e7

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
    ax1.set_title(f'THEMIS-{probe} ESA Ion Differential Energy Flux & Orbit ({date_title})',
                  fontsize=12, fontweight='bold', pad=12)

    cbar1 = fig.colorbar(pcm, ax=ax1, pad=0.02, aspect=20)
    cbar1.set_label('EFlux [eV/(cm² s sr eV)]', fontsize=10, fontweight='bold')

    ax1.set_xlim(x_min, x_max)

    # Adaptive X-axis tick intervals
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

    has_full_state = ('state_lt' in data_dict and 'state_r' in data_dict and len(data_dict['state_lt']) > 0)
    st_times_num = data_dict.get('state_times_num', np.array([]))
    if has_full_state:
        st_lt = data_dict['state_lt']
        st_r = data_dict['state_r']
        lt_unwrapped_st = np.unwrap(st_lt * (2.0 * np.pi / 24.0))

    lt_unwrapped = np.unwrap(lt * (2.0 * np.pi / 24.0))

    for t in ticks:
        if (x_min - 1e-5) <= t <= (x_max + 1e-4):
            if has_full_state and len(st_times_num) > 1 and st_times_num[0] <= t <= st_times_num[-1]:
                r_val = float(np.interp(t, st_times_num, st_r))
                lt_val = float((np.interp(t, st_times_num, lt_unwrapped_st) * 24.0 / (2.0 * np.pi)) % 24.0)
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
        ax1.text(-0.025, -0.05, f'Date:\nUTC:\nR [RE]:\n{coord_label}:', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=8.5, fontweight='bold', linespacing=1.35)
    else:
        ax1.text(-0.025, -0.05, f'UTC:\nR [RE]:\n{coord_label}:', transform=ax1.transAxes,
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
            ax2.set_theta_direction(1)
            hour_labels = ['00 MLT\n(Midnight)', '03', '06 MLT\n(Dawn)', '09',
                           '12 MLT\n(Noon)', '15', '18 MLT\n(Dusk)', '21']
        else:
            ax2.set_theta_zero_location('S')
            ax2.set_theta_direction(1)
            hour_labels = ['00 MLT\n(Midnight)', '03', '06 MLT\n(Dawn)', '09',
                           '12 MLT\n(Noon)', '15', '18 MLT\n(Dusk)', '21']

    # Angular coordinate in radians from local time [0, 24) for downsampled orbit
    theta = lt_orbit * (2.0 * np.pi / 24.0)

    # Continuous trajectory underlay
    if has_full_state:
        theta_st = np.unwrap(st_lt * (2.0 * np.pi / 24.0))
        st_stride = max(1, int(np.ceil(len(st_r) / 4000)))
        ax2.plot(theta_st[::st_stride], st_r[::st_stride], color='#777777', lw=0.6, alpha=0.35, zorder=1, label='Full Orbit Track')
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

    # Dynamic radial limit scaling
    r_candidates = [float(np.nanmax(r_orbit))]
    if has_full_state and len(st_r) > 0:
        r_candidates.append(float(np.nanmax(st_r)))
    peak_r = max(r_candidates)

    if peak_r <= 14.5:
        r_lim = 14.0
        r_ticks = [2, 4, 6, 8, 10, 12, 14]
    elif peak_r <= 20.0:
        r_lim = float(np.ceil(peak_r / 2.0) * 2.0)
        r_ticks = list(range(4, int(r_lim) + 1, 4))
    elif peak_r <= 35.0:
        r_lim = float(np.ceil(peak_r / 5.0) * 5.0)
        r_ticks = list(range(5, int(r_lim) + 1, 5))
    else:
        r_lim = float(np.ceil(peak_r / 10.0) * 10.0)
        r_ticks = list(range(10, int(r_lim) + 1, 10))

    ax2.set_ylim(0, r_lim)
    ax2.set_yticks(r_ticks)
    ax2.set_yticklabels([f'{r} RE' for r in r_ticks], fontsize=8, color='#444444', fontweight='semibold')
    ax2.set_rlabel_position(315)

    # Azimuthal ticks (8 sectors: every 3 hours)
    az_ticks = np.linspace(0, 2.0 * np.pi, 8, endpoint=False)
    ax2.set_xticks(az_ticks)
    ax2.set_xticklabels(hour_labels, fontsize=9, fontweight='bold')

    # Draw Earth boundary circle at center (R <= 1.0 Re)
    theta_earth = np.linspace(0, 2.0 * np.pi, 200)
    ax2.fill(theta_earth, np.ones_like(theta_earth) * 1.0, color='#87CEEB', alpha=0.45, zorder=2)
    ax2.plot(theta_earth, np.ones_like(theta_earth) * 1.0, color='#1E90FF', lw=1.2, zorder=2)
    ax2.text(0, 0, 'Earth\n(1 RE)', ha='center', va='center', fontsize=7, fontweight='bold', color='#104E8B', zorder=4)

    # Mark Start (green square) and End (red triangle) positions
    if has_full_state and len(st_r) > 0:
        ax2.scatter(theta_st[0], st_r[0], marker='s', s=45, color='lime', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Orbit Start')
        ax2.scatter(theta_st[-1], st_r[-1], marker='^', s=55, color='red', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Orbit End')
    else:
        ax2.scatter(theta[0], r_orbit[0], marker='s', s=45, color='lime', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Start')
        ax2.scatter(theta[-1], r_orbit[-1], marker='^', s=55, color='red', edgecolors='black',
                    linewidths=1.2, zorder=5, label='End')

    ax2.grid(True, color='gray', linestyle=':', alpha=0.6)
    if coord == 'gse':
        title_text = f'THEMIS-{probe} Polar Orbit in GSE: R [RE] vs. GSE Local Time (Colored by Ion Flux at {target_energy_keV:.2f} keV)'
    else:
        title_text = f'THEMIS-{probe} Polar Orbit in GSM: R [RE] vs. MLT (Colored by Ion Flux at {target_energy_keV:.2f} keV)'
    ax2.set_title(title_text, fontsize=11.5, fontweight='bold', pad=22)

    # Polar plot colorbar
    cbar2 = fig.colorbar(sc, ax=ax2, pad=0.08, aspect=25, shrink=0.75)
    cbar2.set_label(f'EFlux @ ~{target_energy_keV:.2f} keV [eV/(cm² s sr eV)]', fontsize=9.5, fontweight='bold')

    # Legend in lower right corner
    ax2.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=8.5, framealpha=0.9)

    # Resilient saving with fallback for Windows/OneDrive locks
    print(f"[Plotting] Saving figure to {output_path} (DPI={dpi})...")
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    except OSError as err:
        print(f"[Plotting] Warning: Direct write to {output_path} hit file lock ({err}). Using atomic temp write...")
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f'themis_{int(time.time())}.png')
        plt.savefig(temp_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        for attempt in range(5):
            try:
                shutil.copy2(temp_file, output_path)
                os.remove(temp_file)
                print(f"[Plotting] Successfully saved to {output_path} on attempt {attempt + 1}.")
                return
            except OSError:
                time.sleep(0.5)
        print(f"[Plotting] Warning: Could not overwrite {output_path}. Image saved at {temp_file}")
        return

    plt.close(fig)
    print(f"[Plotting] Successfully saved plot to {output_path}.")


# =============================================================================
# 6. Main CLI Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot THEMIS ESA Ion Spectrogram alongside Spacecraft Orbit Trajectory (R vs Local Time)."
    )
    parser.add_argument('--probe', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'],
                        help="THEMIS Spacecraft Probe: 'a', 'b', 'c', 'd', 'e' (default: 'a').")
    parser.add_argument('--date', nargs='+', default=None,
                        help="Date(s) to analyze: single date (e.g. 2018-01-01) or start and end dates (e.g. 2018-01-01 2018-01-03).")
    parser.add_argument('--trange', nargs=2, default=None,
                        help="Time range as two timestamps: start and end (e.g. 2018-01-01 2018-01-04).")
    parser.add_argument('--coord', type=str, default='gse', choices=['gse', 'gsm'],
                        help="Coordinate system: 'gse' (default, smooth solar-ecliptic orbit) or 'gsm' (geomagnetic).")
    parser.add_argument('--energy', type=float, default=1.0,
                        help="Target ion energy slice in keV for polar orbit trajectory coloring (default: 1.0 keV).")
    parser.add_argument('--mode', type=str, default='reduced', choices=['reduced', 'full'],
                        help="ESA ion mode: 'reduced' (peir, continuous spin survey, default) or 'full' (peif).")
    parser.add_argument('--noon', type=str, default='top', choices=['top', 'bottom'],
                        help="Orientation of polar orbit plot: 'top' (Sun/Noon at top, default) or 'bottom'.")
    parser.add_argument('--point-size', type=float, default=None,
                        help="Custom scatter point size for the polar orbit plot curve (e.g., 0.8, 1.2, 2.0).")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Custom scatter point opacity (alpha) on polar orbit plot (default: 1.0, fully opaque / no transparency).")
    parser.add_argument('--interval', '--cadence', dest='interval', type=float, default=15.0,
                        help="Time interval in minutes to sample polar orbit points (default: 15.0 min). Set 0 for all points without limitation.")
    parser.add_argument('--step', type=int, default=None,
                        help="Optional manual step/stride to downsample orbit points (e.g. --step 5 draws every 5th point).")
    parser.add_argument('--download', action='store_true',
                        help="Auto-download missing CDF files if not cached locally.")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Custom local directory path to search for CDF files.")
    parser.add_argument('--output', type=str, default=None,
                        help="Output image path (default: themis_{probe}_esa_ion_orbit.png).")
    parser.add_argument('--dpi', type=int, default=300,
                        help="Output image resolution DPI (default: 300).")

    args = parser.parse_args()

    probe = args.probe.lower()
    raw_dates = args.trange if args.trange else (args.date if args.date else ['2018-01-01', '2018-01-03'])
    dt_start, dt_end, calendar_dates = parse_date_arguments(raw_dates)

    if args.output is None:
        out_name = f'themis_{probe}_esa_ion_orbit.png'
    else:
        out_name = args.output

    interval_str = f"{args.interval:g} min (no point limit)" if (args.interval and args.interval > 0) else "None (Full points)"
    print("=" * 78)
    print(f"THEMIS-{probe.upper()} ESA Ion Spectrogram & Polar Orbit Trajectory Analysis")
    print(f"Time Range : {dt_start.strftime('%Y-%m-%d %H:%M:%S')} to {dt_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Calendar Days : {len(calendar_dates)} day(s) -> {calendar_dates[0]} to {calendar_dates[-1]}")
    print(f"Coordinates: {args.coord.upper()} | Energy Slice: {args.energy:.2f} keV | ESA Mode: {args.mode}")
    print(f"Noon Axis  : {args.noon.capitalize()} | Output: {out_name}")
    print(f"Orbit Cadence: {interval_str} | Step={args.step if args.step else 'None'}")
    print("=" * 78)

    # 1. Rapidly discover local CDF files (or download if explicitly enabled)
    esa_files, state_files = find_local_themis_files(
        calendar_dates=calendar_dates,
        probe=probe,
        data_dir=args.data_dir,
        allow_download=args.download
    )

    # 2. Parse and interpolate data
    data_dict = load_themis_esa_and_orbit(
        esa_files=esa_files,
        state_files=state_files,
        probe=probe,
        coord=args.coord,
        mode=args.mode,
        clip_start_dt=dt_start,
        clip_end_dt=dt_end
    )

    # 3. Generate visualization
    plot_themis_esa_and_orbit(
        data_dict=data_dict,
        target_energy_keV=args.energy,
        requested_start_dt=dt_start,
        requested_end_dt=dt_end,
        noon_loc=args.noon,
        point_size=args.point_size,
        alpha=args.alpha,
        interval_min=args.interval,
        step=args.step,
        output_path=out_name,
        dpi=args.dpi
    )

    print("=" * 78)
    print(f"Done! Plot saved successfully to: {out_name}")
    print("=" * 78)


if __name__ == '__main__':
    main()

