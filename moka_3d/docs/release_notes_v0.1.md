# MOKA<sup>3D</sup> v0.1
Initial public release.

## Features

- 3D kinematic modeling of disk and outflow components
- YAML-based configuration
- CLI workflow for validation and execution (`validate` + `run`)
- Diagnostic plots and FITS outputs

## Documentation

- README
- Quickstart
- Configuration Guide
- Output Interpretation Guide

## Recent updates
- Mar 25, 2026 -- Added enclosed dynamical mass profile for 'disk' and 'disk_then_outflow' fit
- Mar 26, 2026 -- Added enclosed density profile for 'disk' and 'disk_then_outflow' fit
- Mar 31, 2026 -- Added FITS table output of per-shell kinematic properties (v, v_err, β, β_err) after grid-search fitting

# MOKA<sup>3D</sup> v0.2

- Jun 03, 2026 -- Added radius info to FITS table output of per-shell kinematic properties after grid-search fitting 
- Jun 03, 2026 -- Added PV diagram output .png for pure disc fit mode
- Jun 03, 2026 -- Added radial profile output plot for outflow energetic properties
- Jun 03, 2026 -- Added possibility to give in input either a 2D electron density map or M/L conversion factor map to derive outflow energetic properties 
