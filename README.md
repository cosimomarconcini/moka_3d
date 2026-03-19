# MOKA3D

MOKA3D is a scientific Python package for 3D kinematic modeling of emission-line gas in spectral cubes. It enables fitting of rotating disks, outflows, and combined disk+outflow systems directly in cube space, producing both kinematic diagnostics and derived physical properties.

## Key Features

- Fit rotating disks, outflows, and combined disk+outflow systems from spectral cubes.
- Work directly on FITS data cubes with wavelength- or frequency-based spectral axes.
- Use a reproducible command-line workflow driven by YAML configuration files.
- Produce summary tables, diagnostic plots, model cubes, moment maps, and optional energetics outputs.

## Installation

The installable Python package lives in the `moka3d/` subdirectory of this repository.

Clone the repository:

```bash
git clone https://github.com/cosimomarconcini/moka3d.git
cd moka3d
```

### pip

From the repository root:

```bash
python -m pip install -e ./moka3d
```

If you prefer a non-editable install:

```bash
python -m pip install ./moka3d
```

### Optional conda

If you want an isolated environment, the simplest option is to create it from the bundled `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate moka3d
```

If you prefer to create the environment manually:

```bash
conda create -n moka3d_env python=3.11
conda activate moka3d_env
python -m pip install -e ./moka3d
```

Verify that the CLI is available:

```bash
moka3d --help
```

## Quickstart

The example configuration uses relative paths such as `./Data`, so you must run it from inside the `moka3d/` subdirectory.

Run the example end-to-end:

```bash
git clone https://github.com/cosimomarconcini/moka3d.git
cd moka3d
python -m pip install -e ./moka3d

# Move into the example directory (required because of relative paths)
cd moka3d

# Check installation
moka3d --help

# Write a template config for future runs
moka3d init-config example_config.yaml

# Validate the bundled example config
moka3d validate config_moka.yaml

# Run the bundled example
moka3d run config_moka.yaml
```

The example uses:

- `config_moka.yaml`
- sample cubes in `Data/`
- ancillary maps in `Ancillary_material/`

When the run completes, a new timestamped directory will appear under:

```text
Outputs/YYYY-MM-DD_HHMMSS_<cube_name>/
```

This directory contains:

- `summary.json` - machine-readable best-fit parameters and run summary
- `moka3d.log` - execution log with warnings and runtime details
- diagnostic plots (`.png`) - masks, residuals, shell diagnostics, map comparisons
- optional FITS outputs - model cubes, moment maps, and derived products

## Documentation

- [Quickstart](docs/quickstart.md)
- [Configuration Guide](docs/configuration.md)
- [Outputs Guide](docs/outputs.md)
- [Troubleshooting](docs/troubleshooting.md)

## Known Limitations

MOKA3D is actively evolving. Current limitations include:

- Runtime can grow substantially with large parameter grids, many shells, and high Monte Carlo sampling.
- Energetics are limited to supported emission lines and require a valid FITS `BUNIT`.
- The CLI and YAML configuration define the stable public interface; low-level Python APIs should be considered internal.

## Citation

If you use MOKA3D in scientific work, cite the associated methodological paper and the software release you used. A machine-readable citation file such as `CITATION.cff` should be included in the first public release.

## Release Notes

- [MOKA3D v0.1](docs/release_notes_v0.1.md)
