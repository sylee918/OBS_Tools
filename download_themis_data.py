#!/usr/bin/env python3
"""
download_themis_data.py
=======================
Dedicated data downloader for THEMIS (Time History of Events and Macroscale Interactions
during Substorms) satellite mission.
Separates network data acquisition from data visualization and analysis.

Features:
  1. Ultra-fast 1-pass local cache verification (< 0.5s).
     Skips remote queries for dates that already have local ESA and State CDF files.
  2. Batch downloading across custom date ranges.
  3. Supports all 5 probes: THEMIS-A (tha) through THEMIS-E (the).
  4. Supports both Reduced survey ('peir') and Full survey ('peif') modes.

Usage Examples:
  # 1. Download THEMIS-A data for 3 days:
  python download_themis_data.py --probe a --date 2018-01-01 2018-01-03

  # 2. Download 1 month of THEMIS-D data:
  python download_themis_data.py --probe d --trange 2018-01-01 2018-02-01

  # 3. Force re-download:
  python download_themis_data.py --probe a --date 2018-01-01 --force
"""

import os
import sys
import argparse
import datetime
import glob
import warnings

warnings.filterwarnings('ignore')


def parse_date_arguments(date_args):
    if not date_args:
        date_args = ['2018-01-01', '2018-01-03']
    if isinstance(date_args, str):
        date_args = date_args.split()

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

    if len(date_args) == 1:
        dt_start = parse_single_str(date_args[0])
        dt_end = dt_start + datetime.timedelta(days=1)
    else:
        dt_start = parse_single_str(date_args[0])
        dt_end = parse_single_str(date_args[1])
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


def build_themis_file_index(search_dirs, probe='a'):
    """
    Scans search directories in a single pass and organizes discovered files by date key (YYYYMMDD).
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

                # th{probe}_l2_esa_YYYYMMDD_v*.cdf
                if fname.startswith(esa_prefix):
                    parts = fname[len(esa_prefix):].split('_')
                    if parts and len(parts[0]) >= 8:
                        date_key = parts[0][:8]
                        esa_index.setdefault(date_key, []).append(fpath)

                # th{probe}_l1_state_YYYYMMDD_v*.cdf
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


def download_themis_batch(probe, trange_chunk, mode='reduced', get_esa=True, get_state=True):
    import pyspedas.projects.themis as themis

    t_start = trange_chunk[0].strftime('%Y-%m-%d')
    t_end = (trange_chunk[1] + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    trange_str = [t_start, t_end]

    downloaded = {'esa': [], 'state': []}

    if get_state:
        try:
            res = themis.state(trange=trange_str, probe=probe, notplot=True, time_clip=False)
            if res:
                downloaded['state'].extend(res)
        except Exception as e:
            print(f"    [Warning] THEMIS state download error for {trange_str}: {e}")

    if get_esa:
        try:
            varnames = [f'th{probe}_peir_en_eflux', f'th{probe}_peir_density'] if mode == 'reduced' else None
            res = themis.esa(trange=trange_str, probe=probe, level='l2', notplot=True,
                             time_clip=False, varnames=varnames)
            if res:
                downloaded['esa'].extend(res)
        except Exception as e:
            print(f"    [Warning] THEMIS ESA download error for {trange_str}: {e}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Dedicated Downloader for THEMIS Satellite ESA and State Data."
    )
    parser.add_argument('--probe', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'],
                        help="THEMIS Spacecraft Probe ('a', 'b', 'c', 'd', 'e'). Default: 'a'.")
    parser.add_argument('--date', nargs='+', default=None,
                        help="Single date or multi-day range (e.g. 2018-01-01 2018-01-03).")
    parser.add_argument('--trange', nargs=2, default=None,
                        help="Time range strings: start and end (e.g. 2018-01-01 2018-02-01).")
    parser.add_argument('--mode', type=str, default='reduced', choices=['reduced', 'full'],
                        help="ESA ion mode: 'reduced' (default) or 'full'.")
    parser.add_argument('--datatype', type=str, default='both', choices=['both', 'esa', 'state'],
                        help="Data type to download: 'both' (default), 'esa', or 'state'.")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Custom local directory to store/check downloaded files.")
    parser.add_argument('--batch-days', type=int, default=14,
                        help="Days per batch request (default: 14).")
    parser.add_argument('--force', action='store_true',
                        help="Force re-download even if files already exist locally.")

    args = parser.parse_args()

    probe = args.probe.lower()
    raw_dates = args.trange if args.trange else (args.date if args.date else ['2018-01-01', '2018-01-03'])
    dt_start, dt_end, calendar_dates = parse_date_arguments(raw_dates)

    get_esa = args.datatype in ['both', 'esa']
    get_state = args.datatype in ['both', 'state']

    search_dirs = []
    if args.data_dir:
        search_dirs.append(os.path.normpath(args.data_dir))
    search_dirs.extend([
        os.path.normpath(f'themis_data/th{probe}'),
        os.path.normpath('themis_data'),
        os.path.normpath(f'pydata/themis/th{probe}'),
        os.path.normpath('pydata/themis')
    ])

    print("=" * 78)
    print(f" THEMIS-{probe.upper()} Data Downloader (CDAWeb / SPEDAS)")
    print(f" Target Period  : {dt_start.strftime('%Y-%m-%d')} to {dt_end.strftime('%Y-%m-%d')} ({len(calendar_dates)} calendar days)")
    print(f" Data Types     : ESA={'Yes' if get_esa else 'No'}, State={'Yes' if get_state else 'No'}")
    print(f" ESA Mode       : {args.mode}")
    print(f" Batch Size     : {args.batch_days} days per network request")
    print(f" Force Download : {args.force}")
    print("=" * 78)

    # 1. 1-pass local cache check
    print(f"[Cache Check] Scanning local directories for existing THEMIS-{probe.upper()} files...")
    t0_scan = datetime.datetime.now()
    esa_index, state_index = build_themis_file_index(search_dirs, probe=probe)
    scan_sec = (datetime.datetime.now() - t0_scan).total_seconds()
    print(f"[Cache Check] Completed in {scan_sec:.2f}s! Found {sum(len(v) for v in esa_index.values())} ESA files across {len(esa_index)} dates, "
          f"{sum(len(v) for v in state_index.values())} State files across {len(state_index)} dates.")

    # 2. Determine missing dates
    missing_dates = []
    cached_dates_count = 0

    for d_str in calendar_dates:
        dt = datetime.datetime.strptime(d_str, '%Y-%m-%d')
        date_compact = dt.strftime('%Y%m%d')

        has_esa = (date_compact in esa_index and len(esa_index[date_compact]) > 0) if get_esa else True
        has_state = (date_compact in state_index and len(state_index[date_compact]) > 0) if get_state else True

        if args.force or not (has_esa and has_state):
            missing_dates.append(dt)
        else:
            cached_dates_count += 1

    print(f"[Cache Status] {cached_dates_count}/{len(calendar_dates)} dates already cached locally. "
          f"{len(missing_dates)} dates need downloading.")

    if not missing_dates:
        print("\n[Complete] All requested data is already downloaded and cached locally!")
        print(f"You can immediately run:")
        print(f"  python plot_themis_esa_proton_orbit.py --probe {probe} --date {calendar_dates[0]} {calendar_dates[-1]}\n")
        return

    # 3. Group into batches
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

    total_downloaded_esa = 0
    total_downloaded_state = 0

    for idx, (b_start, b_end) in enumerate(batches, 1):
        span_days = (b_end - b_start).days + 1
        print(f"\n[{idx}/{len(batches)}] Downloading {b_start.strftime('%Y-%m-%d')} to {b_end.strftime('%Y-%m-%d')} ({span_days} days)...")
        res = download_themis_batch(probe, (b_start, b_end), mode=args.mode, get_esa=get_esa, get_state=get_state)
        n_esa = len(res.get('esa', []))
        n_st = len(res.get('state', []))
        total_downloaded_esa += n_esa
        total_downloaded_state += n_st
        print(f"    -> Batch {idx} finished: retrieved {n_esa} ESA files, {n_st} State files.")

    print("\n" + "=" * 78)
    print(" Download Completed Successfully!")
    print(f" Total files fetched: ESA={total_downloaded_esa}, State={total_downloaded_state}")
    print("=" * 78)
    print(f"\nYou can now visualize this interval with:")
    print(f"  python plot_themis_esa_proton_orbit.py --probe {probe} --date {calendar_dates[0]} {calendar_dates[-1]}\n")


if __name__ == '__main__':
    main()

