#!/usr/bin/env python3
"""
save_themis_esa_proton_orbit_nc.py
==================================
Extracts THEMIS (THA, THB, THC, THD, THE) ESA ion differential energy flux and
spacecraft State orbit coordinates (Time, X, Y, Z, R, LT) from local CDF files
and saves them into a standardized NetCDF (.nc) file across user-specified
energy channels (default: 12 energy levels matching MMS FPI analysis).

Supports both single-core and multi-core parallel CDF reading (--workers N).

Default Energy Grid (keV):
  [0.010, 0.015, 0.029, 0.055, 0.110, 0.209, 0.439, 0.872, 1.821, 2.730, 4.290, 6.000]

Variables Saved in .nc:
  - time: Unix timestamp in seconds (float64, 1D)
  - x, y, z: Spacecraft position in Earth Radii (R_E, float64, 1D)
  - r: Radial distance in Earth Radii (R_E, float64, 1D)
  - lt: Local Time / MLT in hours [0, 24) (float64, 1D)
  - energy: Energy grid in keV (float64, 1D, length=N_energies)
  - flux: 2D array of differential energy flux (time, energy) [keV/(cm^2 s sr keV)]
  - flux_EXX_<energy>keV: Individual 1D flux variables for each energy channel

Usage Examples:
  python3 save_themis_esa_proton_orbit_nc.py --probe a --date 2018-01-01 2018-12-31 --workers 36 --output tha_orbit_flux_2018.nc
  python3 save_themis_esa_proton_orbit_nc.py --probe a --date 2018-01-01 2018-01-03

Author: Space Physics Pair-Programming Assistant
"""

import os
import argparse
import datetime
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cdflib
from scipy.interpolate import interp1d

# Support both netCDF4 and scipy.io.netcdf fallback
try:
    import netCDF4 as nc
    HAS_NETCDF4 = True
except ImportError:
    from scipy.io import netcdf_file
    HAS_NETCDF4 = False

DEFAULT_ENERGIES = [
    0.010, 0.015, 0.029, 0.055, 0.110, 0.209,
    0.439, 0.872, 1.821, 2.730, 4.290, 6.000
]


def normalize_themis_probe(probe_arg):
    """
    Normalizes probe identifier ('a', 'tha', 'THEMIS-A', etc.) to single lowercase letter 'a'-'e'.
    """
    if not probe_arg:
        return 'a'
    p = str(probe_arg).strip().lower()
    for prefix in ('themis-', 'themis_', 'themis', 'th-', 'th_', 'th'):
        if p.startswith(prefix):
            p = p[len(prefix):]
    p = p.strip()
    return p if p in ['a', 'b', 'c', 'd', 'e'] else 'a'


def parse_date_or_trange(date_args, trange_args=None):
    raw_args = trange_args if (trange_args and len(trange_args) > 0) else date_args
    if not raw_args:
        raw_args = ['2018-01-01', '2018-01-03']

    if isinstance(raw_args, str):
        raw_args = raw_args.split()

    def parse_single_str(s):
        s = s.strip()
        for fmt in ('%Y-%m-%d/%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d/%H:%M',
                    '%Y-%m-%d %H:%M', '%Y-%m-%d', '%Y%m%d'):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                pass
        raise ValueError(f"Unrecognized date/time format: '{s}'")

    if len(raw_args) == 1:
        dt0 = parse_single_str(raw_args[0])
        dt_start = dt0.replace(hour=0, minute=0, second=0)
        dt_end = dt0.replace(hour=23, minute=59, second=59)
    else:
        dt_start = parse_single_str(raw_args[0])
        dt_end = parse_single_str(raw_args[-1])
        if dt_end.hour == 0 and dt_end.minute == 0 and dt_end.second == 0 and len(raw_args[-1].strip()) <= 10:
            dt_end = dt_end.replace(hour=23, minute=59, second=59)

    if dt_end < dt_start:
        dt_start, dt_end = dt_end, dt_start

    calendar_dates = []
    curr = dt_start.date()
    while curr <= dt_end.date():
        calendar_dates.append(curr.strftime('%Y-%m-%d'))
        curr += datetime.timedelta(days=1)

    return dt_start, dt_end, calendar_dates


def build_themis_file_index(search_dirs, probe='a'):
    """
    Single-pass directory scan to index local THEMIS ESA (L2) and State (L1) CDF files by date (YYYYMMDD).
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


def find_local_themis_files(calendar_dates, probe='a', data_dir=None):
    """
    Locates local THEMIS ESA and State CDF files across standard directory trees.
    """
    probe = normalize_themis_probe(probe)
    search_dirs = []
    if data_dir:
        search_dirs.append(os.path.normpath(data_dir))
    search_dirs.extend([
        os.path.normpath(f'themis_data/th{probe}'),
        os.path.normpath('themis_data'),
        os.path.normpath(f'pydata/themis/th{probe}'),
        os.path.normpath('pydata/themis'),
        os.path.normpath('.')
    ])

    print("[File Discovery] Rapidly indexing local THEMIS directories...", flush=True)
    esa_index, state_index = build_themis_file_index(search_dirs, probe=probe)

    esa_files = []
    state_files = []
    missing_dates = []

    for d_str in calendar_dates:
        dc = d_str.replace('-', '')
        e_list = esa_index.get(dc, [])
        s_list = state_index.get(dc, [])
        if e_list and s_list:
            esa_files.extend(e_list)
            state_files.extend(s_list)
        else:
            missing_dates.append(d_str)
            esa_files.extend(e_list)
            state_files.extend(s_list)

    esa_files = sorted(list(set(esa_files)))
    state_files = sorted(list(set(state_files)))

    print(f"[File Discovery] Found {len(esa_files)} ESA files and {len(state_files)} State files.", flush=True)
    if missing_dates:
        print(f"[File Discovery] Note: {len(missing_dates)} dates missing locally: {missing_dates[:5]}...", flush=True)

    return esa_files, state_files


def _init_multiprocessing_worker():
    """
    Worker initializer for multiprocessing.Pool.
    Unpins CPU affinity mask on HPC PBS / Slurm clusters so workers utilize all allocated cores.
    """
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        try:
            total_cpus = os.cpu_count() or 36
            for n in range(total_cpus, 0, -1):
                try:
                    os.sched_setaffinity(0, set(range(n)))
                    break
                except OSError:
                    continue
        except Exception:
            pass


def unbind_process_affinity(requested_workers=None):
    """
    Unbinds CPU affinity mask in the parent process if restricted to a single core by PBS.
    """
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        try:
            curr = os.sched_getaffinity(0)
            total = os.cpu_count() or 36
            print(f"[Affinity] Initial CPU affinity mask: {len(curr)} core(s) out of {total} detected.", flush=True)
            if len(curr) < total:
                for n in range(total, 0, -1):
                    try:
                        os.sched_setaffinity(0, set(range(n)))
                        new_aff = os.sched_getaffinity(0)
                        print(f"[Affinity] Expanded CPU affinity to {len(new_aff)} cores.", flush=True)
                        break
                    except OSError:
                        continue
        except Exception as e:
            print(f"[Affinity] Note: sched affinity check skipped: {e}", flush=True)


def extract_flux_at_energy(energy_eV, flux_2d, target_energy_eV):
    """
    Fast log-log interpolation across energy channels for each time step.
    Handles descending energy tables and trailing NaN channels (e.g. 24 active bins out of 32 in THEMIS ESA).
    energy_eV: shape (N, 32) or (32,)
    flux_2d: shape (N, 32)
    target_energy_eV: scalar float in eV
    """
    n_times = flux_2d.shape[0]
    flux_target = np.full(n_times, np.nan, dtype=np.float64)
    log_target_e = np.log10(target_energy_eV)

    if energy_eV.ndim == 1:
        energy_2d = np.tile(energy_eV, (n_times, 1))
    else:
        energy_2d = energy_eV

    # Replace NaNs in energy table with -1.0 before np.unique so NaN columns group together properly
    e_clean = np.where(np.isfinite(energy_2d) & (energy_2d > 0), np.round(energy_2d, decimals=2), -1.0)
    unique_e, inverse_indices = np.unique(e_clean, axis=0, return_inverse=True)

    if len(unique_e) < 100:
        for u_i, u_grid in enumerate(unique_e):
            mask = (inverse_indices == u_i)
            valid_e = (u_grid > 0)
            if np.sum(valid_e) < 2:
                continue
            sub_e = u_grid[valid_e]
            s = np.argsort(sub_e)
            sub_e_sorted = sub_e[s]
            if target_energy_eV < sub_e_sorted[0] or target_energy_eV > sub_e_sorted[-1]:
                continue

            log_e = np.log10(sub_e_sorted)
            idx = int(np.searchsorted(sub_e_sorted, target_energy_eV))
            f_sub = flux_2d[mask][:, valid_e][:, s]

            if 0 < idx < len(sub_e_sorted):
                e0, e1 = log_e[idx - 1], log_e[idx]
                f0 = f_sub[:, idx - 1]
                f1 = f_sub[:, idx]
                valid_f = np.isfinite(f0) & np.isfinite(f1) & (f0 > 0) & (f1 > 0)
                t = (log_target_e - e0) / (e1 - e0) if (e1 != e0) else 0.0
                res = np.full(f_sub.shape[0], np.nan, dtype=np.float64)
                res[valid_f] = 10.0 ** ((1.0 - t) * np.log10(f0[valid_f]) + t * np.log10(f1[valid_f]))

                # Fallback for rows where one bracketing channel was NaN but other channels exist
                partial = (~valid_f) & (np.sum(np.isfinite(f_sub) & (f_sub > 0), axis=1) >= 2)
                if np.any(partial):
                    for r_idx in np.where(partial)[0]:
                        frow = f_sub[r_idx]
                        vf = np.isfinite(frow) & (frow > 0)
                        res[r_idx] = 10.0 ** np.interp(log_target_e, log_e[vf], np.log10(frow[vf]))

                flux_target[mask] = res
            elif idx == 0 and np.isclose(target_energy_eV, sub_e_sorted[0]):
                f0 = f_sub[:, 0]
                flux_target[mask] = np.where(np.isfinite(f0) & (f0 > 0), f0, np.nan)
            elif idx == len(sub_e_sorted) and np.isclose(target_energy_eV, sub_e_sorted[-1]):
                f1 = f_sub[:, -1]
                flux_target[mask] = np.where(np.isfinite(f1) & (f1 > 0), f1, np.nan)
        return flux_target

    # Row-by-row fallback
    for i in range(n_times):
        e_row = energy_2d[i]
        f_row = flux_2d[i]
        valid = np.isfinite(e_row) & np.isfinite(f_row) & (f_row > 0) & (e_row > 0)
        if np.sum(valid) >= 2:
            e_sub = e_row[valid]
            f_sub = f_row[valid]
            s = np.argsort(e_sub)
            flux_target[i] = 10.0 ** np.interp(log_target_e, np.log10(e_sub[s]), np.log10(f_sub[s]))

    return flux_target


def _parse_single_esa_file(args):
    """
    Worker function to parse a single THEMIS ESA L2 CDF file.
    """
    f, probe, mode = args
    prefix = f'th{probe}_'
    mode_tag = 'peir' if mode == 'reduced' else 'peif'
    time_var_candidates = [f'{prefix}{mode_tag}_time', f'{prefix}peir_time', f'{prefix}peif_time']
    flux_var_candidates = [f'{prefix}{mode_tag}_en_eflux', f'{prefix}peir_en_eflux', f'{prefix}peif_en_eflux']
    yaxis_var_candidates = [f'{prefix}{mode_tag}_en_eflux_yaxis', f'{prefix}peir_en_eflux_yaxis', f'{prefix}peif_en_eflux_yaxis']

    try:
        cdf = cdflib.CDF(f)
        zvars = set(cdf.cdf_info().zVariables)

        t_var = next((v for v in time_var_candidates if v in zvars), None)
        fl_var = next((v for v in flux_var_candidates if v in zvars), None)
        y_var = next((v for v in yaxis_var_candidates if v in zvars), None)
        if not t_var or not fl_var or not y_var:
            return None

        raw_t = cdf.varget(t_var)
        raw_fl = cdf.varget(fl_var)
        raw_y = cdf.varget(y_var)
        if raw_t is None or raw_fl is None or raw_y is None or np.size(raw_t) == 0:
            return None

        u_sec = np.atleast_1d(np.asarray(raw_t, dtype=np.float64))
        fl = np.asarray(raw_fl, dtype=np.float64)
        if fl.ndim == 1:
            fl = fl.reshape(1, -1)
        elif fl.ndim == 0 or fl.size == 0:
            return None

        y_arr = np.asarray(raw_y, dtype=np.float64)
        if y_arr.ndim == 1:
            en_file = np.tile(y_arr, (len(u_sec), 1))
        elif y_arr.ndim == 2:
            if y_arr.shape[0] == len(u_sec):
                en_file = y_arr
            else:
                en_file = np.tile(y_arr[0], (len(u_sec), 1))
        else:
            return None

        if len(u_sec) != fl.shape[0] or len(u_sec) != en_file.shape[0]:
            return None

        return (u_sec, fl, en_file)
    except Exception:
        return None


def _parse_single_state_file(args):
    """
    Worker function to parse a single THEMIS State L1 CDF file.
    """
    f, probe, coord = args
    pos_var_name = f'th{probe}_pos_gse' if coord == 'gse' else f'th{probe}_pos_gsm'
    time_var_name = f'th{probe}_state_time'

    try:
        cdf = cdflib.CDF(f)
        zvars = set(cdf.cdf_info().zVariables)

        t_var = next((v for v in [time_var_name, f'th{probe}_state_time', 'tha_state_time'] if v in zvars), None)
        p_var = next((v for v in [pos_var_name, f'th{probe}_pos_gse', f'th{probe}_pos_gsm'] if v in zvars), None)
        if not t_var or not p_var:
            return None

        raw_t = cdf.varget(t_var)
        raw_p = cdf.varget(p_var)
        if raw_t is None or raw_p is None or np.size(raw_t) == 0:
            return None

        u_sec = np.atleast_1d(np.asarray(raw_t, dtype=np.float64))
        p_vec = np.asarray(raw_p, dtype=np.float64)
        if p_vec.ndim == 1:
            p_vec = p_vec.reshape(1, -1)
        if p_vec.shape[1] != 3:
            return None

        bad_p = np.abs(p_vec[:, 0]) > 1e10
        p_vec[bad_p] = np.nan

        # Normalize km -> Earth Radii (6371.2 km)
        x_re = p_vec[:, 0] / 6371.2
        y_re = p_vec[:, 1] / 6371.2
        z_re = p_vec[:, 2] / 6371.2
        r_re = np.linalg.norm(p_vec, axis=1) / 6371.2

        phi = np.arctan2(p_vec[:, 1], p_vec[:, 0])
        lt_val = (phi * 12.0 / np.pi + 12.0) % 24.0

        if len(u_sec) != len(r_re) or len(u_sec) != len(lt_val):
            return None

        return (u_sec, x_re, y_re, z_re, r_re, lt_val)
    except Exception:
        return None


def load_and_process_themis_data(esa_files, state_files, target_energies_keV,
                                 probe='a', coord='gse', mode='reduced',
                                 clip_start_dt=None, clip_end_dt=None,
                                 workers=None):
    """
    Loads THEMIS ESA spectra and State ephemeris (in parallel if workers > 1),
    interpolates coordinates onto ESA timestamps, and extracts flux slices.
    """
    if not esa_files:
        raise FileNotFoundError(f"No THEMIS-{probe.upper()} ESA files available.")
    if not state_files:
        raise FileNotFoundError(f"No THEMIS-{probe.upper()} State files available.")

    if workers is None:
        workers = min(36, os.cpu_count() or 4)
    workers = max(1, int(workers))

    if workers > 1:
        unbind_process_affinity(workers)

    esa_epochs_list = []
    esa_flux_list = []
    esa_energy_list = []

    print(f"[Parser] Reading {len(esa_files)} THEMIS-{probe.upper()} ESA files with {workers} worker(s)...", flush=True)
    esa_tasks = [(f, probe, mode) for f in esa_files]
    if workers > 1 and len(esa_tasks) > 1:
        chunksize = max(1, len(esa_tasks) // (workers * 4))
        with multiprocessing.Pool(processes=workers, initializer=_init_multiprocessing_worker) as pool:
            results = pool.map(_parse_single_esa_file, esa_tasks, chunksize=chunksize)
    else:
        results = [_parse_single_esa_file(t) for t in esa_tasks]

    for res in results:
        if res is not None:
            esa_epochs_list.append(res[0])
            esa_flux_list.append(res[1])
            esa_energy_list.append(res[2])

    if not esa_epochs_list:
        raise ValueError("Could not extract valid ESA data from provided CDF files.")

    esa_epoch = np.concatenate(esa_epochs_list)
    esa_flux = np.concatenate(esa_flux_list, axis=0)
    esa_energy = np.concatenate(esa_energy_list, axis=0)

    s_idx = np.argsort(esa_epoch)
    esa_epoch = esa_epoch[s_idx]
    esa_flux = esa_flux[s_idx]
    esa_energy = esa_energy[s_idx]

    esa_flux[esa_flux <= 0] = np.nan
    esa_flux[esa_flux >= 1e25] = np.nan

    # Parse State files
    coord = coord.lower()
    st_epochs_list = []
    st_x_list = []
    st_y_list = []
    st_z_list = []
    st_r_list = []
    st_lt_list = []

    print(f"[Parser] Reading {len(state_files)} THEMIS-{probe.upper()} State files with {workers} worker(s)...", flush=True)
    st_tasks = [(f, probe, coord) for f in state_files]
    if workers > 1 and len(st_tasks) > 1:
        chunksize = max(1, len(st_tasks) // (workers * 4))
        with multiprocessing.Pool(processes=workers, initializer=_init_multiprocessing_worker) as pool:
            st_results = pool.map(_parse_single_state_file, st_tasks, chunksize=chunksize)
    else:
        st_results = [_parse_single_state_file(t) for t in st_tasks]

    for res in st_results:
        if res is not None:
            st_epochs_list.append(res[0])
            st_x_list.append(res[1])
            st_y_list.append(res[2])
            st_z_list.append(res[3])
            st_r_list.append(res[4])
            st_lt_list.append(res[5])

    if not st_epochs_list:
        raise ValueError("Could not extract valid State data from provided CDF files.")

    all_st_epoch = np.concatenate(st_epochs_list)
    all_st_x = np.concatenate(st_x_list)
    all_st_y = np.concatenate(st_y_list)
    all_st_z = np.concatenate(st_z_list)
    all_st_r = np.concatenate(st_r_list)
    all_st_lt = np.concatenate(st_lt_list)

    m_sort = np.argsort(all_st_epoch)
    all_st_epoch = all_st_epoch[m_sort]
    all_st_x = all_st_x[m_sort]
    all_st_y = all_st_y[m_sort]
    all_st_z = all_st_z[m_sort]
    all_st_r = all_st_r[m_sort]
    all_st_lt = all_st_lt[m_sort]

    # Interpolate State orbit onto ESA timestamps
    print(f"[Parser] Interpolating orbit parameters onto {len(esa_epoch):,} ESA timestamps...", flush=True)
    valid_m = np.isfinite(all_st_epoch) & np.isfinite(all_st_r) & np.isfinite(all_st_lt)
    ep_m = all_st_epoch[valid_m]

    f_x = interp1d(ep_m, all_st_x[valid_m], bounds_error=False, fill_value='extrapolate')
    f_y = interp1d(ep_m, all_st_y[valid_m], bounds_error=False, fill_value='extrapolate')
    f_z = interp1d(ep_m, all_st_z[valid_m], bounds_error=False, fill_value='extrapolate')
    f_r = interp1d(ep_m, all_st_r[valid_m], bounds_error=False, fill_value='extrapolate')

    lt_rad = np.unwrap(all_st_lt[valid_m] * (2.0 * np.pi / 24.0))
    f_lt = interp1d(ep_m, lt_rad, bounds_error=False, fill_value='extrapolate')

    interp_x = f_x(esa_epoch)
    interp_y = f_y(esa_epoch)
    interp_z = f_z(esa_epoch)
    interp_r = f_r(esa_epoch)
    interp_lt = (f_lt(esa_epoch) * 24.0 / (2.0 * np.pi)) % 24.0

    # Optional time clipping
    if clip_start_dt or clip_end_dt:
        t_mask = np.ones(len(esa_epoch), dtype=bool)
        if clip_start_dt:
            t_mask &= (esa_epoch >= clip_start_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        if clip_end_dt:
            t_mask &= (esa_epoch <= clip_end_dt.replace(tzinfo=datetime.timezone.utc).timestamp())

        esa_epoch = esa_epoch[t_mask]
        esa_flux = esa_flux[t_mask]
        esa_energy = esa_energy[t_mask]
        interp_x = interp_x[t_mask]
        interp_y = interp_y[t_mask]
        interp_z = interp_z[t_mask]
        interp_r = interp_r[t_mask]
        interp_lt = interp_lt[t_mask]

    # Extract flux slices for all requested energy levels
    print(f"[Parser] Extracting flux slices for {len(target_energies_keV)} energy channels...", flush=True)
    n_energies = len(target_energies_keV)
    flux_matrix = np.full((len(esa_epoch), n_energies), np.nan, dtype=np.float64)

    for idx, e_kev in enumerate(target_energies_keV):
        target_eV = e_kev * 1000.0
        flux_matrix[:, idx] = extract_flux_at_energy(esa_energy, esa_flux, target_eV)

    return {
        'time': esa_epoch,
        'x': interp_x,
        'y': interp_y,
        'z': interp_z,
        'r': interp_r,
        'lt': interp_lt,
        'energies_keV': np.array(target_energies_keV, dtype=np.float64),
        'flux': flux_matrix,
        'coord': coord.upper(),
        'probe': probe.upper()
    }


def save_to_netcdf(data_dict, output_nc_path):
    """
    Saves processed THEMIS orbit and flux data into a standard NetCDF (.nc) file.
    """
    out_dir = os.path.dirname(os.path.abspath(output_nc_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t = data_dict['time']
    x = data_dict['x']
    y = data_dict['y']
    z = data_dict['z']
    r = data_dict['r']
    lt = data_dict['lt']
    energies = data_dict['energies_keV']
    flux = data_dict['flux']
    coord = data_dict['coord']
    probe = data_dict['probe']

    n_times = len(t)
    n_energies = len(energies)

    if HAS_NETCDF4:
        print(f"[NetCDF] Writing NetCDF4 file to {output_nc_path} (via netCDF4)...", flush=True)
        with nc.Dataset(output_nc_path, 'w', format='NETCDF4') as ds:
            ds.title = f"THEMIS-{probe} ESA Ion Flux and Orbit Trajectory"
            ds.spacecraft = f"THA" if probe == 'A' else f"TH{probe}"
            ds.coordinate_system = coord
            ds.created = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            ds.description = "Spacecraft orbit coordinates (X, Y, Z, R, LT) and ion flux at selected energy channels"

            ds.createDimension('time', n_times)
            ds.createDimension('energy', n_energies)

            v_time = ds.createVariable('time', 'f8', ('time',), zlib=True)
            v_time.units = 'seconds since 1970-01-01 00:00:00 UTC'
            v_time.long_name = 'Unix Timestamp'
            v_time[:] = t

            v_x = ds.createVariable('x', 'f8', ('time',), zlib=True)
            v_x.units = 'R_E'
            v_x.long_name = f'X Position in {coord}'
            v_x[:] = x

            v_y = ds.createVariable('y', 'f8', ('time',), zlib=True)
            v_y.units = 'R_E'
            v_y.long_name = f'Y Position in {coord}'
            v_y[:] = y

            v_z = ds.createVariable('z', 'f8', ('time',), zlib=True)
            v_z.units = 'R_E'
            v_z.long_name = f'Z Position in {coord}'
            v_z[:] = z

            v_r = ds.createVariable('r', 'f8', ('time',), zlib=True)
            v_r.units = 'R_E'
            v_r.long_name = 'Radial Distance from Earth Center'
            v_r[:] = r

            v_lt = ds.createVariable('lt', 'f8', ('time',), zlib=True)
            v_lt.units = 'hours'
            v_lt.long_name = f'Local Time ({coord})'
            v_lt[:] = lt

            v_en = ds.createVariable('energy', 'f8', ('energy',), zlib=True)
            v_en.units = 'keV'
            v_en.long_name = 'Ion Energy Channel'
            v_en[:] = energies

            v_flux = ds.createVariable('flux', 'f8', ('time', 'energy'), zlib=True)
            v_flux.units = 'keV / (cm^2 s sr keV)'
            v_flux.long_name = 'Omnidirectional Ion Differential Energy Flux'
            v_flux[:] = flux

            for idx, e_val in enumerate(energies):
                var_name = f"flux_E{idx+1:02d}_{str(e_val).replace('.', 'p')}keV"
                v_1d = ds.createVariable(var_name, 'f8', ('time',), zlib=True)
                v_1d.units = 'keV / (cm^2 s sr keV)'
                v_1d.long_name = f'Ion Differential Energy Flux at {e_val} keV'
                v_1d[:] = flux[:, idx]
    else:
        print(f"[NetCDF] Writing NetCDF3 file to {output_nc_path} (via scipy.io.netcdf)...", flush=True)
        with netcdf_file(output_nc_path, 'w') as ds:
            ds.title = f"THEMIS-{probe} ESA Ion Flux and Orbit Trajectory"
            ds.spacecraft = f"TH{probe}"
            ds.coordinate_system = coord
            ds.created = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

            ds.createDimension('time', n_times)
            ds.createDimension('energy', n_energies)

            for name, arr, units in [
                ('time', t, 'seconds since 1970-01-01 00:00:00 UTC'),
                ('x', x, 'R_E'),
                ('y', y, 'R_E'),
                ('z', z, 'R_E'),
                ('r', r, 'R_E'),
                ('lt', lt, 'hours'),
            ]:
                v = ds.createVariable(name, 'f8', ('time',))
                v.units = units
                v[:] = arr

            v_en = ds.createVariable('energy', 'f8', ('energy',))
            v_en.units = 'keV'
            v_en[:] = energies

            v_flux = ds.createVariable('flux', 'f8', ('time', 'energy'))
            v_flux.units = 'keV / (cm^2 s sr keV)'
            v_flux[:] = flux

            for idx, e_val in enumerate(energies):
                var_name = f"flux_E{idx+1:02d}_{str(e_val).replace('.', 'p')}keV"
                v_1d = ds.createVariable(var_name, 'f8', ('time',))
                v_1d[:] = flux[:, idx]

    print(f"[NetCDF] Successfully created: {output_nc_path}", flush=True)
    print(f"         Points: {n_times:,} timestamps, {n_energies} energy levels.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract THEMIS ESA Ion Flux & State Orbit to NetCDF (.nc) file."
    )
    parser.add_argument('--probe', type=str, default='a',
                        help="THEMIS Probe letter ('a', 'b', 'c', 'd', 'e'). Default: a")
    parser.add_argument('--date', nargs='*', default=['2018-01-01', '2018-01-03'],
                        help="Single date or multi-day range (e.g. '2018-01-01 2018-12-31').")
    parser.add_argument('--trange', nargs='*', default=None,
                        help="Time range strings (e.g. '2018-01-01 2018-12-31').")
    parser.add_argument('--coord', type=str, default='gse', choices=['gse', 'gsm', 'GSE', 'GSM'],
                        help="Coordinate system: 'gse' or 'gsm'. Default: 'gse'.")
    parser.add_argument('--mode', type=str, default='reduced', choices=['reduced', 'full'],
                        help="ESA data mode: 'reduced' (peir, default) or 'full' (peif).")
    parser.add_argument('--energies', nargs='*', type=float, default=DEFAULT_ENERGIES,
                        help="Custom list of energy levels in keV.")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Custom local directory path for THEMIS CDF files.")
    parser.add_argument('--workers', '--ncpus', dest='workers', type=int, default=None,
                        help="Number of parallel worker processes for CDF reading (default: min(36, cpu_count)).")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output .nc file path (default: th{probe}_esa_orbit_{start}_{end}.nc).")

    args = parser.parse_args()
    probe = normalize_themis_probe(args.probe)
    dt_start, dt_end, calendar_dates = parse_date_or_trange(args.date, args.trange)

    if args.output is None:
        start_str = dt_start.strftime('%Y%m%d')
        end_str = dt_end.strftime('%Y%m%d')
        out_name = (
            f"th{probe}_esa_orbit_{start_str}_{end_str}.nc"
            if start_str != end_str
            else f"th{probe}_esa_orbit_{start_str}.nc"
        )
    else:
        out_name = args.output

    print("=" * 76, flush=True)
    print(f" THEMIS-{probe.upper()} ESA & State Orbit NetCDF Data Extractor", flush=True)
    print(f" Time Range : {dt_start.strftime('%Y-%m-%d %H:%M:%S')} to {dt_end.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f" Coordinate : {args.coord.upper()} (ESA mode: {args.mode})", flush=True)
    print(f" Energies   : {len(args.energies)} channels: {args.energies}", flush=True)
    print(f" Workers    : {args.workers if args.workers else 'Auto (up to 36 CPUs)'}", flush=True)
    print(f" Output .nc : {out_name}", flush=True)
    print("=" * 76, flush=True)

    esa_files, state_files = find_local_themis_files(calendar_dates, probe=probe, data_dir=args.data_dir)

    data_dict = load_and_process_themis_data(
        esa_files=esa_files,
        state_files=state_files,
        target_energies_keV=args.energies,
        probe=probe,
        coord=args.coord.lower(),
        mode=args.mode.lower(),
        clip_start_dt=dt_start,
        clip_end_dt=dt_end,
        workers=args.workers
    )

    save_to_netcdf(data_dict, out_name)


if __name__ == '__main__':
    main()

