<h1 style="display: flex; align-items: flex-start; gap: 10px;">
  <img src="moka_3d/logo_moka3d.png" width="80" height="100">
  MOKA<sup>3D</sup>
</h1>

MOKA<sup>3D</sup> is a scientific Python package for 3D kinematic modeling of emission-line gas in spectral cubes. It enables fitting of rotating disks, outflows, and combined disk+outflow systems directly in cube space, producing both kinematic diagnostics and derived physical properties.

## Key Features

- Fit rotating disks, outflows, and combined disk+outflow systems from spectral cubes.
- Work directly on FITS data cubes with wavelength- or frequency-based spectral axes.
- Use a reproducible command-line workflow driven by YAML configuration files.
- Produce summary tables, diagnostic plots, model cubes, moment maps, and optional energetics outputs.

## Installation

The installable Python package lives in the `moka3d/` subdirectory of this repository.

Clone the repository:

```bash
git clone https://github.com/cosimomarconcini/moka_3d.git
cd moka_3d/moka_3d/
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
git clone https://github.com/cosimomarconcini/moka_3d.git
cd moka_3d/moka_3d/
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
- ancillary SN maps in `Ancillary_material/`

When the run starts, a new timestamped directory will appear under:

```text
Outputs/YYYY-MM-DD_HHMMSS_<cube_name>/
```

This directory contains:

- `summary.json` - machine-readable best-fit parameters and run summary
- `moka3d.log` - execution log with warnings and runtime details
- diagnostic plots (`.png`) - masks, residuals, shell diagnostics, map comparisons
- optional FITS outputs - model cubes, moment maps, and derived products

## Documentation

- [Quickstart](moka_3d/docs/quickstart.md)
- [Configuration Guide](moka_3d/docs/configuration.md)
- [Outputs Guide](moka_3d/docs/outputs.md)

## Known Limitations

MOKA<sup>3D</sup> is actively evolving. Current limitations include:

- Runtime can grow substantially with large parameter grids, many shells, and high Monte Carlo sampling.
- The outflow energetics are limited to supported emission lines ($\mathrm{H}\beta$, $[\mathrm{O III}] 5007\\mathrm{Å}$, $\mathrm{H}\alpha$) and require a valid FITS `BUNIT`.
- The CLI and YAML configuration define the stable public interface; low-level Python APIs should be considered internal.

## Citation

If you use MOKA<sup>3D</sup> for your research, please cite cite the associated methodological paper [Marconcini+23](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A..58M/abstract), with the following BibTeX entry and the software release you used:

```bibtex
@ARTICLE{2023A&A...677A..58M,
       author = {{Marconcini}, C. and {Marconi}, A. and {Cresci}, G. and {Venturi}, G. and {Ulivi}, L. and {Mannucci}, F. and {Belfiore}, F. and {Tozzi}, G. and {Ginolfi}, M. and {Marasco}, A. and {Carniani}, S. and {Amiri}, A. and {Di Teodoro}, E. and {Scialpi}, M. and {Tomicic}, N. and {Mingozzi}, M. and {Brazzini}, M. and {Moreschini}, B.},
        title = "{MOKA$^{3D}$: An innovative approach to 3D gas kinematic modelling. I. Application to AGN ionised outflows}",
      journal = {\aap},
     keywords = {galaxies: Seyfert, galaxies: kinematics and dynamics, galaxies: active, ISM: jets and outflows, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = sep,
       volume = {677},
          eid = {A58},
        pages = {A58},
          doi = {10.1051/0004-6361/202346821},
archivePrefix = {arXiv},
       eprint = {2307.01854},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...677A..58M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Release Notes

- [MOKA3D v0.1](moka_3d/docs/release_notes_v0.1.md)

## MOKA<sup>3D</sup> updates
To stay updated on MOKA<sup>3D</sup> most recent releases, features, and tutorials join the mailing list:

[MOKA<sup>3D</sup> mailing list: ]([https://choosealicense.com/licenses/gpl-3.0/](https://docs.google.com/forms/d/e/1FAIpQLSc9MKXQRyQo5GCDNHGVNM8v8mGoLpBChD8QSnbXpyQBi5wxKw/viewform)) 
 


## License
This project is licensed under the terms of the [GNU General Public License version 3.0](https://choosealicense.com/licenses/gpl-3.0/) license.

## Disclaimer

This software is provided for research purposes only.
The authors make no guarantees regarding the correctness, reliability, or suitability of the results produced.

Any use of this software, including in scientific publications, is at your own risk. The authors are not responsible for any incorrect results, conclusions, or damages arising from its use.

