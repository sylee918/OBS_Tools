#!/usr/bin/env python3
"""
save_themis_esa_proton_orbit_nc_parallel.py
===========================================
Parallel HPC entry point for extracting THEMIS (THA-THE) ESA ion flux and
State orbit coordinates into a NetCDF (.nc) file using multiprocessing workers.
"""

from save_themis_esa_proton_orbit_nc import main

if __name__ == '__main__':
    main()

