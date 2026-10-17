#!/usr/bin/env python3
"""
save_mms1_fpi_proton_orbit_nc.py
================================
Extracts MMS FPI fast survey ion differential energy flux and spacecraft orbit coordinates
(Time, X, Y, Z, R, LT) and saves them into a NetCDF (.nc) file across user-specified
energy levels (default: 12 energy levels).

Default Energy Grid (keV):
  [0.010, 0.015, 0.029, 0.055, 0.110, 0.209, 0.439, 0.872, 1.821, 2.730, 4.290, 6.000]

Variables Saved in .nc:
  - time: Unix timestamp in seconds (float64, 1D)
  - x, y, z: Spacecraft position in Earth Radii (R_E, float64, 1D)
  - r: Radial distance in Earth Radii (R_E, float64, 1D)
  - lt: Local Time / MLT in hours [0, 24) (float64, 1D)
  - energy: Energy grid in keV (float64, 1D, length=N_energies)
  - flux: 2D array of differential energy flux (time, energy) [keV/(cm² s sr keV)]
  - flux_<energy>keV: Individual 1D flux variables for each energy channel

Author: Space Physics Pair-Programming Assistant
"""

import os
import re
import argparse
import datetime
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


def normalize_probe_name(probe_arg):
    if not probe_arg:
        return '1'
    p = str(probe_arg).strip().lower().replace('mms', '').replace('-', '')
    return p if p in ['1', '2', '3', '4'] else '1'


def parse_date_or_trange(date_args, trange_args=None):
    raw_args = trange_args if (trange_args and len(trange_args) > 0) else date_args
    if not raw_args:
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


def find_local_mms_files(calendar_dates, probe='1', data_dir=None):
    search_dirs = []
    if data_dir:
        search_dirs.append(os.path.normpath(data_dir))
    search_dirs.extend([
        os.path.normpath(f'pydata/mms{probe}'),
        os.path.normpath(f'mms_data/mms{probe}'),
        os.path.normpath('mms_data'),
        os.path.normpath('pydata')
    ])

    print(f"[File Discovery] Rapidly indexing local directories...")
    fpi_index, mec_index = build_mms_file_index(search_dirs, probe=probe)

    fpi_files = []
    mec_files = []
    missing_dates = []

    for d_str in calendar_dates:
        date_compact = d_str.replace('-', '')
        if date_compact in fpi_index and len(fpi_index[date_compact]) > 0:
            fpi_files.extend(fpi_index[date_compact])
        else:
            missing_dates.append(d_str)

        if date_compact in mec_index and len(mec_index[date_compact]) > 0:
            mec_files.extend(mec_index[date_compact])

    fpi_files = sorted(list(dict.fromkeys(fpi_files)))
    mec_files = sorted(list(dict.fromkeys(mec_files)))

    print(f"[File Discovery] Found {len(fpi_files)} FPI files and {len(mec_files)} MEC files.")
    if missing_dates:
        print(f"[File Discovery] Note: {len(missing_dates)} dates not found locally: {missing_dates[:5]}...")

    return fpi_files, mec_files


def extract_flux_at_energy(energy_eV, flux_2d, target_energy_eV):
    """
    Log-log interpolation across energy channels for each time step.
    Supports time-varying energy tables (e.g. Survey mode vs Solar Wind mode).
    energy_eV: shape (N, 32) or (32,)
    flux_2d: shape (N, 32)
    """
    n_times = flux_2d.shape[0]
    flux_target = np.full(n_times, np.nan, dtype=np.float64)
    log_target_e = np.log10(target_energy_eV)
    is_2d = (energy_eV.ndim == 2)

    for i in range(n_times):
        e_row = energy_eV[i] if is_2d else energy_eV
        f_row = flux_2d[i]
        valid = np.isfinite(e_row) & np.isfinite(f_row) & (f_row > 0) & (e_row > 0)
        if np.sum(valid) >= 2:
            e_sub = e_row[valid]
            f_sub = f_row[valid]
            s = np.argsort(e_sub)
            flux_target[i] = 10.0 ** np.interp(log_target_e, np.log10(e_sub[s]), np.log10(f_sub[s]))

    return flux_target


def load_and_process_mms_data(fpi_files, mec_files, target_energies_keV,
                              probe='1', coord='gse', clip_start_dt=None, clip_end_dt=None):
    """
    Loads FPI spectra and MEC ephemeris, interpolates coordinates onto FPI timestamps,
    and extracts flux slices across all target_energies_keV.
    """
    if not fpi_files:
        raise FileNotFoundError(f"No MMS{probe} FPI files available.")
    if not mec_files:
        raise FileNotFoundError(f"No MMS{probe} MEC files available.")

    print(f"[Parser] Reading {len(fpi_files)} FPI files...")
    fpi_epochs_list = []
    fpi_flux_list = []
    fpi_energy_list = []

    for f in fpi_files:
        try:
            cdf = cdflib.CDF(f)
            zvars = set(cdf.cdf_info().zVariables)
            ep_var = next((v for v in ['Epoch', 'Epoch_plus_var', 'epoch'] if v in zvars), None)
            fl_var = next((v for v in [
                f'mms{probe}_dis_energyspectr_omni_fast',
                'mms1_dis_energyspectr_omni_fast',
                f'mms{probe}_dis_energyspectr_fast',
                'mms1_dis_energyspectr_fast'
            ] if v in zvars), None)
            if not ep_var or not fl_var:
                continue

            raw_ep = cdf.varget(ep_var)
            raw_fl = cdf.varget(fl_var)
            if raw_ep is None or raw_fl is None or np.size(raw_ep) == 0:
                continue

            u_sec = np.atleast_1d(cdflib.cdfepoch.unixtime(raw_ep))
            fl = np.asarray(raw_fl, dtype=np.float64)
            if fl.ndim == 1:
                fl = fl.reshape(1, -1)
            elif fl.ndim == 0 or fl.size == 0:
                continue

            en_var = next((v for v in [
                f'mms{probe}_dis_energy_fast',
                'mms1_dis_energy_fast',
                f'mms{probe}_dis_energy_omni_fast',
                'mms1_dis_energy_omni_fast'
            ] if v in zvars), None)
            if not en_var:
                continue
            raw_en = cdf.varget(en_var)
            if raw_en is None or np.size(raw_en) == 0:
                continue
            raw_en = np.asarray(raw_en, dtype=np.float64)
            if raw_en.ndim == 1:
                en_file = np.tile(raw_en, (len(u_sec), 1))
            elif raw_en.ndim == 2:
                if raw_en.shape[0] == len(u_sec):
                    en_file = raw_en
                else:
                    en_file = np.tile(raw_en[0], (len(u_sec), 1))
            else:
                continue

            if len(u_sec) != fl.shape[0] or len(u_sec) != en_file.shape[0]:
                print(f"[Parser] Warning: Skipping {os.path.basename(f)} due to mismatch: epoch={len(u_sec)}, flux={fl.shape[0]}, en={en_file.shape[0]}")
                continue

            fpi_epochs_list.append(u_sec)
            fpi_flux_list.append(fl)
            fpi_energy_list.append(en_file)
        except Exception as e:
            print(f"[Parser] Warning skipping {f}: {e}")

    if not fpi_epochs_list:
        raise ValueError("Could not extract valid FPI data from provided CDF files.")

    fpi_epoch = np.concatenate(fpi_epochs_list)
    fpi_flux = np.concatenate(fpi_flux_list, axis=0)
    fpi_energy = np.concatenate(fpi_energy_list, axis=0)

    s_idx = np.argsort(fpi_epoch)
    fpi_epoch = fpi_epoch[s_idx]
    fpi_flux = fpi_flux[s_idx]
    fpi_energy = fpi_energy[s_idx]

    fpi_flux[fpi_flux <= 0] = np.nan
    fpi_flux[fpi_flux >= 1e25] = np.nan

    # Parse MEC
    coord = coord.lower()
    mec_epochs_list = []
    mec_x_list = []
    mec_y_list = []
    mec_z_list = []
    mec_r_list = []
    mec_lt_list = []

    print(f"[Parser] Reading {len(mec_files)} MEC files (coord: {coord.upper()})...")
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
            r_vec = np.asarray(r_raw, dtype=np.float64)
            if r_vec.ndim == 1:
                r_vec = r_vec.reshape(1, -1)
            if r_vec.shape[1] != 3:
                continue

            bad_r = np.abs(r_vec[:, 0]) > 1e10
            r_vec[bad_r] = np.nan

            # Earth radius normalization (6371.2 km)
            x_re = r_vec[:, 0] / 6371.2
            y_re = r_vec[:, 1] / 6371.2
            z_re = r_vec[:, 2] / 6371.2
            r_re = np.linalg.norm(r_vec, axis=1) / 6371.2

            if coord == 'gse':
                phi = np.arctan2(r_vec[:, 1], r_vec[:, 0])
                lt_val = (phi * 12.0 / np.pi + 12.0) % 24.0
            else:
                mlt_var = next((v for v in [f'mms{probe}_mec_mlt', 'mms1_mec_mlt'] if v in zvars), None)
                if mlt_var:
                    raw_mlt = cdf.varget(mlt_var)
                    if raw_mlt is not None and np.size(raw_mlt) > 0:
                        lt_val = np.asarray(raw_mlt, dtype=np.float64)
                        lt_val[(lt_val < 0) | (lt_val > 24)] = np.nan
                    else:
                        phi = np.arctan2(r_vec[:, 1], r_vec[:, 0])
                        lt_val = (phi * 12.0 / np.pi + 12.0) % 24.0
                else:
                    phi = np.arctan2(r_vec[:, 1], r_vec[:, 0])
                    lt_val = (phi * 12.0 / np.pi + 12.0) % 24.0

            mec_epochs_list.append(u_sec)
            mec_x_list.append(x_re)
            mec_y_list.append(y_re)
            mec_z_list.append(z_re)
            mec_r_list.append(r_re)
            mec_lt_list.append(lt_val)
        except Exception as e:
            print(f"[Parser] Warning skipping {f}: {e}")

    all_mec_epoch = np.concatenate(mec_epochs_list)
    all_mec_x = np.concatenate(mec_x_list)
    all_mec_y = np.concatenate(mec_y_list)
    all_mec_z = np.concatenate(mec_z_list)
    all_mec_r = np.concatenate(mec_r_list)
    all_mec_lt = np.concatenate(mec_lt_list)

    m_sort = np.argsort(all_mec_epoch)
    all_mec_epoch = all_mec_epoch[m_sort]
    all_mec_x = all_mec_x[m_sort]
    all_mec_y = all_mec_y[m_sort]
    all_mec_z = all_mec_z[m_sort]
    all_mec_r = all_mec_r[m_sort]
    all_mec_lt = all_mec_lt[m_sort]

    # Interpolate MEC orbit onto FPI timestamps
    print(f"[Parser] Interpolating orbit parameters onto {len(fpi_epoch)} FPI timestamps...")
    valid_m = np.isfinite(all_mec_epoch) & np.isfinite(all_mec_r) & np.isfinite(all_mec_lt)
    ep_m = all_mec_epoch[valid_m]

    f_x = interp1d(ep_m, all_mec_x[valid_m], bounds_error=False, fill_value='extrapolate')
    f_y = interp1d(ep_m, all_mec_y[valid_m], bounds_error=False, fill_value='extrapolate')
    f_z = interp1d(ep_m, all_mec_z[valid_m], bounds_error=False, fill_value='extrapolate')
    f_r = interp1d(ep_m, all_mec_r[valid_m], bounds_error=False, fill_value='extrapolate')

    lt_rad = np.unwrap(all_mec_lt[valid_m] * (2.0 * np.pi / 24.0))
    f_lt = interp1d(ep_m, lt_rad, bounds_error=False, fill_value='extrapolate')

    interp_x = f_x(fpi_epoch)
    interp_y = f_y(fpi_epoch)
    interp_z = f_z(fpi_epoch)
    interp_r = f_r(fpi_epoch)
    interp_lt = (f_lt(fpi_epoch) * 24.0 / (2.0 * np.pi)) % 24.0

    # Optional time clipping
    if clip_start_dt or clip_end_dt:
        t_mask = np.ones(len(fpi_epoch), dtype=bool)
        if clip_start_dt:
            t_mask &= (fpi_epoch >= clip_start_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        if clip_end_dt:
            t_mask &= (fpi_epoch <= clip_end_dt.replace(tzinfo=datetime.timezone.utc).timestamp())

        fpi_epoch = fpi_epoch[t_mask]
        fpi_flux = fpi_flux[t_mask]
        fpi_energy = fpi_energy[t_mask]
        interp_x = interp_x[t_mask]
        interp_y = interp_y[t_mask]
        interp_z = interp_z[t_mask]
        interp_r = interp_r[t_mask]
        interp_lt = interp_lt[t_mask]

    # Extract flux slices for all requested energy levels
    print(f"[Parser] Extracting flux slices for {len(target_energies_keV)} energy channels (handling dynamic energy tables)...")
    n_energies = len(target_energies_keV)
    flux_matrix = np.full((len(fpi_epoch), n_energies), np.nan, dtype=np.float64)

    for idx, e_kev in enumerate(target_energies_keV):
        target_eV = e_kev * 1000.0
        flux_matrix[:, idx] = extract_flux_at_energy(fpi_energy, fpi_flux, target_eV)

    return {
        'time': fpi_epoch,
        'x': interp_x,
        'y': interp_y,
        'z': interp_z,
        'r': interp_r,
        'lt': interp_lt,
        'energies_keV': np.array(target_energies_keV, dtype=np.float64),
        'flux': flux_matrix,
        'coord': coord.upper(),
        'probe': probe
    }


def save_to_netcdf(data_dict, output_nc_path):
    """
    Saves processed orbit and flux data into a standard NetCDF (.nc) file.
    Supports netCDF4 (with compression) or scipy.io.netcdf fallback.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_nc_path)), exist_ok=True)
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
        print(f"[NetCDF] Writing NetCDF4 file to {output_nc_path} (via netCDF4)...")
        with nc.Dataset(output_nc_path, 'w', format='NETCDF4') as ds:
            # Global attributes
            ds.title = f"MMS{probe} FPI Fast Survey Ion Flux and Orbit Trajectory"
            ds.spacecraft = f"MMS{probe}"
            ds.coordinate_system = coord
            ds.created = datetime.datetime.utcnow().isoformat() + "Z"
            ds.description = "Spacecraft orbit coordinates (X, Y, Z, R, LT) and ion flux at selected energy channels"

            # Dimensions
            ds.createDimension('time', n_times)
            ds.createDimension('energy', n_energies)

            # Variables
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
            v_r.long_name = 'Geocentric Radial Distance'
            v_r[:] = r

            v_lt = ds.createVariable('lt', 'f8', ('time',), zlib=True)
            v_lt.units = 'hours'
            v_lt.long_name = f'Local Time / MLT in {coord}'
            v_lt[:] = lt

            v_en = ds.createVariable('energy', 'f8', ('energy',))
            v_en.units = 'keV'
            v_en.long_name = 'Ion Energy Channels'
            v_en[:] = energies

            v_flux = ds.createVariable('flux', 'f8', ('time', 'energy'), zlib=True)
            v_flux.units = 'keV/(cm^2 s sr keV)'
            v_flux.long_name = 'Differential Energy Flux'
            v_flux[:] = flux

            # Also create 1D convenience variables for each energy channel
            for idx, e_val in enumerate(energies):
                var_name = f"flux_E{idx+1:02d}_{str(e_val).replace('.', 'p')}keV"
                v_1d = ds.createVariable(var_name, 'f8', ('time',), zlib=True)
                v_1d.units = 'keV/(cm^2 s sr keV)'
                v_1d.long_name = f'Differential Energy Flux at ~{e_val:.3f} keV'
                v_1d[:] = flux[:, idx]
    else:
        print(f"[NetCDF] Writing NetCDF-3 file to {output_nc_path} (via scipy.io.netcdf)...")
        with netcdf_file(output_nc_path, 'w') as ds:
            ds.title = f"MMS{probe} FPI Fast Survey Ion Flux and Orbit Trajectory"
            ds.coordinate_system = coord

            ds.createDimension('time', n_times)
            ds.createDimension('energy', n_energies)

            v_time = ds.createVariable('time', 'f8', ('time',))
            v_time[:] = t

            v_x = ds.createVariable('x', 'f8', ('time',))
            v_x[:] = x

            v_y = ds.createVariable('y', 'f8', ('time',))
            v_y[:] = y

            v_z = ds.createVariable('z', 'f8', ('time',))
            v_z[:] = z

            v_r = ds.createVariable('r', 'f8', ('time',))
            v_r[:] = r

            v_lt = ds.createVariable('lt', 'f8', ('time',))
            v_lt[:] = lt

            v_en = ds.createVariable('energy', 'f8', ('energy',))
            v_en[:] = energies

            v_flux = ds.createVariable('flux', 'f8', ('time', 'energy'))
            v_flux[:] = flux

            for idx, e_val in enumerate(energies):
                var_name = f"flux_E{idx+1:02d}_{str(e_val).replace('.', 'p')}keV"
                v_1d = ds.createVariable(var_name, 'f8', ('time',))
                v_1d[:] = flux[:, idx]

    print(f"[NetCDF] Successfully created: {output_nc_path}")
    print(f"         Points: {n_times} timestamps, {n_energies} energy levels.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MMS1 FPI Ion Flux & Orbit to NetCDF (.nc) file."
    )
    parser.add_argument('--probe', type=str, default='1',
                        help="MMS Probe number (1, 2, 3, 4). Default: 1")
    parser.add_argument('--date', nargs='*', default=['2015-12-30', '2015-12-31'],
                        help="Single date or multi-day range (e.g. '2015-12-30 2015-12-31').")
    parser.add_argument('--trange', nargs='*', default=None,
                        help="Time range strings (e.g. '2015-12-30 2016-01-01').")
    parser.add_argument('--coord', type=str, default='gse', choices=['gse', 'gsm', 'GSE', 'GSM'],
                        help="Coordinate system: 'gse' or 'gsm'. Default: 'gse'.")
    parser.add_argument('--energies', nargs='*', type=float, default=DEFAULT_ENERGIES,
                        help="Custom list of energy levels in keV.")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Custom local directory path for CDF files.")
    parser.add_argument('--output', type=str, default=None,
                        help="Output .nc file path (default: mms{probe}_fpi_orbit_{date}.nc).")

    args = parser.parse_args()
    probe = normalize_probe_name(args.probe)
    dt_start, dt_end, calendar_dates = parse_date_or_trange(args.date, args.trange)

    if args.output is None:
        start_str = dt_start.strftime('%Y%m%d')
        end_str = dt_end.strftime('%Y%m%d')
        out_name = f"mms{probe}_fpi_orbit_{start_str}_{end_str}.nc" if start_str != end_str else f"mms{probe}_fpi_orbit_{start_str}.nc"
    else:
        out_name = args.output

    print("=" * 76)
    print(f" MMS{probe} FPI & Orbit NetCDF Data Extractor")
    print(f" Time Range : {dt_start.strftime('%Y-%m-%d %H:%M:%S')} to {dt_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Coordinate : {args.coord.upper()}")
    print(f" Energies   : {len(args.energies)} channels: {args.energies}")
    print(f" Output .nc : {out_name}")
    print("=" * 76)

    # 1. Locate CDF files
    fpi_files, mec_files = find_local_mms_files(calendar_dates, probe=probe, data_dir=args.data_dir)

    # 2. Extract and interpolate
    data_dict = load_and_process_mms_data(
        fpi_files=fpi_files,
        mec_files=mec_files,
        target_energies_keV=args.energies,
        probe=probe,
        coord=args.coord.lower(),
        clip_start_dt=dt_start,
        clip_end_dt=dt_end
    )

    # 3. Save to NetCDF
    save_to_netcdf(data_dict, out_name)


if __name__ == '__main__':
    main()

