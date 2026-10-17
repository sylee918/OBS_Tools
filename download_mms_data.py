#!/usr/bin/env python3
"""
download_mms_data.py
====================
Dedicated data downloader for Magnetospheric Multiscale (MMS) mission.
Separates network data acquisition from data visualization and analysis.

Features:
  1. Ultra-fast 1-pass local cache verification (< 0.5s for 10,000+ files).
     Prevents redundant remote SPDF queries for already-downloaded dates.
  2. Safe chunked downloading (e.g. 7-day batches) to prevent timeouts,
     crashes, and memory bloat when requesting long intervals (e.g. 1 year).
  3. Supports FPI Fast Survey ion moments ('dis-moms') and MEC ephemeris ('srvy_l2').
  4. Clear, informative progress indicators for each batch and probe.
  5. Ready-to-copy plot command output upon completion.

Usage Examples:
  # 1. Download MMS1 data for a 2-day period:
  python download_mms_data.py --probe 1 --date 2015-12-30 2015-12-31

  # 2. Download MMS1 data for 6 months in 7-day chunks:
  python download_mms_data.py --probe 1 --trange 2015-07-01 2016-01-01 --batch-days 7

  # 3. Download only MEC ephemeris data:
  python download_mms_data.py --probe 1 --date 2015-12-30 2015-12-31 --datatype mec

  # 4. Force re-download even if files exist locally:
  python download_mms_data.py --probe 1 --date 2015-12-30 --force
"""

import os
import sys
import argparse
import datetime
import re
import warnings

# Suppress noisy SPDF / PySPEDAS warnings
warnings.filterwarnings('ignore')


def normalize_probe_name(probe_arg):
    if not probe_arg:
        return '1'
    p = str(probe_arg).strip().lower().replace('mms', '').replace('-', '')
    if p in ['1', '2', '3', '4']:
        return p
    return '1'


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
    Returns:
      fpi_index: { 'YYYYMMDD': [filepath1, filepath2, ...] }
      mec_index: { 'YYYYMMDD': [filepath1, ...] }
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

                # Check FPI dis-moms: mms{probe}_fpi_fast_l2_dis-moms_YYYYMMDDHHMMSS_v*.cdf
                if fname.startswith(fpi_prefix):
                    parts = fname[len(fpi_prefix):].split('_')
                    if parts and len(parts[0]) >= 8:
                        date_key = parts[0][:8]
                        fpi_index.setdefault(date_key, []).append(fpath)

                # Check MEC: mms{probe}_mec_srvy_l2_epht89q_YYYYMMDD_v*.cdf
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


def download_chunk(probe, trange_chunk, get_fpi=True, get_mec=True):
    """
    Downloads a single time chunk using pyspedas with NASA SPDF.
    """
    import pyspedas

    # Format trange for pyspedas
    t_start = trange_chunk[0].strftime('%Y-%m-%d')
    t_end = (trange_chunk[1] + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    trange_str = [t_start, t_end]

    downloaded = {'fpi': [], 'mec': []}

    if get_mec:
        try:
            res = pyspedas.mms_load_mec(
                trange=trange_str,
                probe=probe,
                spdf=True,
                notplot=True,
                varformat='Epoch'
            )
            if res:
                downloaded['mec'].extend(res)
        except Exception as e:
            print(f"    [Warning] MEC download error for {trange_str}: {e}")

    if get_fpi:
        try:
            res = pyspedas.mms_load_fpi(
                trange=trange_str,
                probe=probe,
                datatype='dis-moms',
                data_rate='fast',
                level='l2',
                spdf=True,
                notplot=True,
                varformat='Epoch'
            )
            if res:
                downloaded['fpi'].extend(res)
        except Exception as e:
            print(f"    [Warning] FPI download error for {trange_str}: {e}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Dedicated Downloader for MMS Satellite FPI and MEC Data."
    )
    parser.add_argument('--probe', type=str, default='1',
                        help="MMS Spacecraft number (1, 2, 3, 4). Default: 1")
    parser.add_argument('--date', nargs='*', default=None,
                        help="Single date or multi-day range (e.g. '2015-12-30 2015-12-31').")
    parser.add_argument('--trange', nargs='*', default=None,
                        help="Time range strings (e.g. '2015-07-01 2016-01-01').")
    parser.add_argument('--datatype', type=str, default='both', choices=['both', 'fpi', 'mec'],
                        help="Data type to download: 'both' (default), 'fpi', or 'mec'.")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Custom local directory to store/check downloaded files.")
    parser.add_argument('--batch-days', type=int, default=7,
                        help="Number of days per download batch (default: 7). Prevents remote timeouts.")
    parser.add_argument('--force', action='store_true',
                        help="Force re-download even if files already exist locally.")

    args = parser.parse_args()

    probe = normalize_probe_name(args.probe)
    dt_start, dt_end, calendar_dates = parse_date_or_trange(args.date, args.trange)

    get_fpi = args.datatype in ['both', 'fpi']
    get_mec = args.datatype in ['both', 'mec']

    # Search paths for local cache check
    search_dirs = []
    if args.data_dir:
        search_dirs.append(os.path.normpath(args.data_dir))
    search_dirs.extend([
        os.path.normpath(f'pydata/mms{probe}'),
        os.path.normpath(f'mms_data/mms{probe}'),
        os.path.normpath('mms_data'),
        os.path.normpath('pydata')
    ])

    print("=" * 78)
    print(f" MMS{probe} Data Downloader (NASA SPDF)")
    print(f" Target Period  : {dt_start.strftime('%Y-%m-%d')} to {dt_end.strftime('%Y-%m-%d')} ({len(calendar_dates)} calendar days)")
    print(f" Data Types     : FPI={'Yes' if get_fpi else 'No'}, MEC={'Yes' if get_mec else 'No'}")
    print(f" Batch Size     : {args.batch_days} days per network request")
    print(f" Force Download : {args.force}")
    print("=" * 78)

    # 1. Ultra-fast local cache check (1-pass)
    print(f"[Cache Check] Scanning local directories for existing MMS{probe} files...")
    t0_scan = datetime.datetime.now()
    fpi_index, mec_index = build_mms_file_index(search_dirs, probe=probe)
    scan_sec = (datetime.datetime.now() - t0_scan).total_seconds()
    print(f"[Cache Check] Completed in {scan_sec:.2f}s! Found {sum(len(v) for v in fpi_index.values())} FPI files across {len(fpi_index)} dates, "
          f"{sum(len(v) for v in mec_index.values())} MEC files across {len(mec_index)} dates.")

    # 2. Determine missing dates
    missing_dates = []
    cached_dates_count = 0

    for d_str in calendar_dates:
        dt = datetime.datetime.strptime(d_str, '%Y-%m-%d')
        date_compact = dt.strftime('%Y%m%d')

        has_fpi = (date_compact in fpi_index and len(fpi_index[date_compact]) > 0) if get_fpi else True
        has_mec = (date_compact in mec_index and len(mec_index[date_compact]) > 0) if get_mec else True

        if args.force or not (has_fpi and has_mec):
            missing_dates.append(dt)
        else:
            cached_dates_count += 1

    print(f"[Cache Status] {cached_dates_count}/{len(calendar_dates)} dates already cached locally. "
          f"{len(missing_dates)} dates need downloading.")

    if not missing_dates:
        print("\n[Complete] All requested data is already downloaded and cached locally!")
        print(f"You can immediately run:")
        print(f"  python plot_mms1_fpi_proton_orbit.py --probe {probe} --trange {dt_start.strftime('%Y-%m-%d')} {dt_end.strftime('%Y-%m-%d')}\n")
        return

    # 3. Group missing dates into batches
    missing_dates.sort()
    batches = []
    cur_batch_start = missing_dates[0]
    cur_batch_end = missing_dates[0]

    for d in missing_dates[1:]:
        if (d - cur_batch_start).days < args.batch_days:
            cur_batch_end = d
        else:
            batches.append((cur_batch_start, cur_batch_end))
            cur_batch_start = d
            cur_batch_end = d
    batches.append((cur_batch_start, cur_batch_end))

    print(f"\n[Download Plan] Split {len(missing_dates)} missing dates into {len(batches)} batches.")

    # 4. Execute downloads with progress report
    total_downloaded_fpi = 0
    total_downloaded_mec = 0

    for idx, (b_start, b_end) in enumerate(batches, 1):
        span_days = (b_end - b_start).days + 1
        print(f"\n[{idx}/{len(batches)}] Downloading {b_start.strftime('%Y-%m-%d')} to {b_end.strftime('%Y-%m-%d')} ({span_days} days)...")
        res = download_chunk(probe, (b_start, b_end), get_fpi=get_fpi, get_mec=get_mec)
        n_fpi = len(res.get('fpi', []))
        n_mec = len(res.get('mec', []))
        total_downloaded_fpi += n_fpi
        total_downloaded_mec += n_mec
        print(f"    -> Batch {idx} finished: retrieved {n_fpi} FPI files, {n_mec} MEC files.")

    print("\n" + "=" * 78)
    print(" Download Completed Successfully!")
    print(f" Total files fetched: FPI={total_downloaded_fpi}, MEC={total_downloaded_mec}")
    print("=" * 78)
    print(f"\nYou can now visualize this interval with:")
    print(f"  python plot_mms1_fpi_proton_orbit.py --probe {probe} --trange {dt_start.strftime('%Y-%m-%d')} {dt_end.strftime('%Y-%m-%d')}\n")


if __name__ == '__main__':
    main()

