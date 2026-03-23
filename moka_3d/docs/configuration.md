# Configuration Guide

The YAML configuration file is the main interface to MOKA<sup>3D</sup>.

The MOKA<sup>3D</sup> YAML file controls what data are read, how the cube is interpreted, which kinematic model is fitted, and which outputs are written. In normal use, you edit the YAML file and then run:

```bash
moka3d validate config.yaml
moka3d run config.yaml
```

Start from the bundled example config `config_moka.yaml` or from a fresh template created with:

```bash
moka3d init-config my_config.yaml
```

## Overview

The configuration file is divided into logical sections. You do not need to change every field for every run. In most cases, the important edits are:

- your input cube and ancillary paths
- the target redshift and line definition
- the fit mode (`disk`, `outflow`, or `disk_then_outflow`)
- the parameter grids for the component you want to fit

## Recommended workflow

For normal use, follow this sequence every time:

```bash
# 1. Edit the YAML file

# 2. Check that the config is valid
moka3d validate config.yaml

# 3. Run the pipeline
moka3d run config.yaml
```

## General structure

This section shows the overall structure of the YAML file.

A typical configuration has this structure:

```yaml
paths:
  data_dir: ./Data
  ancillary_dir: ./Ancillary_material
  output_dir: ./Outputs

input:
  cube_file: my_cube.fits
  sn_map: null

target:
  agn_ra: null
  agn_dec: null
  center_mode: flux
  center_xy_manual: null
  redshift: 0.005

line:
  wavelength_line: 5006.8
  wavelength_line_unit: Angstrom
  doppler_convention: null

processing:
  sn_thresh: 3
  nrebin: 1
  pixel_scale_arcsec_manual: null
  psf_sigma: [0.9]
  lsf_sigma: 70.0

fit:
  component_mode: disk_then_outflow
  disc:
    mode: independent
  outflow:
    mask_mode: bicone

advanced:
  npt: 200000
  compute_energetics: false

output:
  save_plots: true
  show_plots: false
  save_summary_json: true
```

You can think of the configuration as three layers:

- data and observational setup
- model and fitting parameters
- output and diagnostics

## Paths and inputs

This part tells MOKA<sup>3D</sup> where to find the data and where to write results.

```yaml
paths:
  data_dir: ./Data
  ancillary_dir: ./Ancillary_material
  output_dir: ./Outputs

input:
  cube_file: NGC7582_test_OIII_smooth.fits
  sn_map: SN_map_NGC7582.fits
  ne_map: null
  ne_outflow: null
  save_all_outputs: true
```

Most important fields:

- `paths.data_dir`
  Directory containing the main FITS cube.
- `paths.ancillary_dir`
  Directory containing optional ancillary maps such as S/N maps or density maps.
- `paths.output_dir`
  Directory where run outputs will be written.
- `input.cube_file`
  The FITS cube to analyze.
- `input.sn_map`
  Optional 2D S/N map used for masking.
- `input.ne_map`
  Optional 2D density map for energetics.
- `input.ne_outflow`
  Optional constant electron density value or values for outflow energetics.
- `input.save_all_outputs`
  Enables writing more output products, including FITS products useful for inspection.

Typical choices:

- Use relative paths if you always run from the same directory.
- Use absolute paths if your data live elsewhere.
- Set `sn_map: null` if you do not want S/N masking.
- Set `ne_map: null` and `ne_outflow: null` unless you plan to compute energetics.
- If you use relative paths, run `moka3d` from the directory that contains the config file and the matching data folders.

For a single outflow cone, `ne_outflow` should contain one value:

```yaml
input:
  ne_outflow: [400]
```

For a bicone, it should contain two values:

```yaml
input:
  ne_outflow: [400, 500]
```

## Target and spectral line

This section defines where the source is centered and which spectral line the cube represents. These values directly affect how the cube is interpreted, so they should be checked carefully before every real run.

```yaml
target:
  agn_ra: 23h18m23.6280s
  agn_dec: -42d22m13.512s
  center_mode: null
  center_xy_manual: null
  redshift: 0.005410

line:
  wavelength_line: 5006.8
  wavelength_line_unit: Angstrom
  doppler_convention: null
```

Most important fields:

- `target.redshift`
  This is required. It is used to interpret the spectral axis and derive physical scales.
- `target.agn_ra`, `target.agn_dec`
  Use these if you want to center the model on known sky coordinates. It can either be in hms or degrees.
- `target.center_mode`
  Common choices are `flux`, `kinematic`, or `null`.
- `target.center_xy_manual`
  Use this if you want to force the center to a specific pixel position.
- `line.wavelength_line`
  Rest-frame wavelength or frequency of the fitted emission line.
- `line.wavelength_line_unit`
  Common values are `Angstrom`, `micron`, or `GHz`.
- `line.doppler_convention`
  Optional override. Leave as `null` in most cases.

Practical guidance:

- Use `redshift` even if the cube is already close to systemic. It is part of the physical interpretation of the cube.
- For optical or near-infrared emission lines, leave `doppler_convention: null`.
- For frequency cubes, leave `doppler_convention: null` unless you explicitly need to override the default behavior.
- If you are unsure about the line settings, copy them from the instrument reduction or the FITS header documentation rather than guessing.

Example for a frequency cube:

```yaml
line:
  wavelength_line: 345.796
  wavelength_line_unit: GHz
  doppler_convention: null
```

## Processing

This section controls masking, cube sampling, and instrumental smoothing assumptions. In most first runs, you only need to confirm that these values are reasonable rather than tune them aggressively.

```yaml
processing:
  sn_thresh: 3
  nrebin: 1
  pixel_scale_arcsec_manual: null
  psf_sigma: 0.9
  lsf_sigma: 70.0
  vel_sigma: 0.0
```

Most important fields:

- `processing.sn_thresh`
  S/N threshold for masking if an S/N map is provided.
- `processing.nrebin`
  Spatial rebin factor. Leave at `1` unless you intentionally want coarser sampling.
- `processing.pixel_scale_arcsec_manual`
  Manual pixel scale override if the FITS WCS is incomplete or unusable.
- `processing.psf_sigma`
  Spatial PSF width used in modeling.
- `processing.lsf_sigma`
  Spectral line-spread width used in modeling.
- `processing.vel_sigma`
  Additional random velocity dispersion term.

Display ranges are optional but useful for consistent plots:

```yaml
processing:
  display_ranges:
    flux:
      mode: percentile
      values: [1, 99]
    vel:
      mode: percentile
      values: [1, 99]
    sig:
      mode: percentile
      values: [1, 99]
```

Typical choices:

- Keep `nrebin: 1` for the first run.
- Use `pixel_scale_arcsec_manual` only if validation or runtime warnings indicate that pixel scale could not be inferred.
- Set `sn_thresh: null` if you want to disable S/N masking entirely.
- Keep `psf_sigma` and `lsf_sigma` close to the values appropriate for your instrument setup.

## Fit configuration

This is the main modeling choice: what component you fit and over which parameter ranges. If you change only one section after the input paths, it is usually this one.

```yaml
fit:
  component_mode: disk_then_outflow
```

Allowed values:

- `disk`
  Fit only the disk component.
- `outflow`
  Fit only the outflow component.
- `disk_then_outflow`
  Fit the disk first, then fit the outflow with the disk included.

This section also contains the parameter grids for the disk and outflow models. In practice:

- choose the component mode first
- then make the corresponding parameter grids physically sensible for your target
- keep the grids modest for the first run


Typical choices:

- Use `disk` if you want to model only rotation.
- Use `outflow` if the cube is dominated by a cone-like outflow and you do not want to fit a disk.
- Use `disk_then_outflow` if both components matter and you want the standard combined workflow.

Examples:

```yaml
fit:
  component_mode: disk
```

```yaml
fit:
  component_mode: outflow
```

```yaml
fit:
  component_mode: disk_then_outflow
```

## Disk model

The disk configuration is used when `component_mode` is `disk` or `disk_then_outflow`. You normally choose the disk model family first, then set a sensible radial range and velocity grid.

```yaml
fit:
  disc:
    mode: independent
    radius_range_arcsec: [0.0, 40.0]
    num_shells: 30
    beta_grid_deg: [50, 100, 5]
```

Most important fields:

- `fit.disc.mode`
  Disk model family.
- `fit.disc.radius_range_arcsec`
  Radial fit range for the disk.
- `fit.disc.num_shells`
  Number of radial shells used in the disk model.
- `fit.disc.beta_grid_deg`
  Grid of disk inclination values in the form `[min, max, step]`.
- `fit.disc.pa_deg`
  Optional fixed position angle.

Available disk modes:

- `independent`
  The simplest choice. Each shell is fit with a velocity grid. Good default starting point.
- `disk_kepler`
  Keplerian rotation. Use when the science case is dominated by central mass.
- `NSC`
  Nuclear star cluster parameterization.
- `Plummer`
  Plummer-like mass model.
- `disk_arctan`
  Arctangent rotation curve. Useful when you want a smooth disk rotation profile.

Typical starting choice:

```yaml
fit:
  disc:
    mode: independent
    independent:
      v_grid_kms: [0.0, 400.0, 20.0]
```

Arctangent example:

```yaml
fit:
  disc:
    mode: disk_arctan
    arctan:
      rt_arcsec: 1.5
      vmax_grid_kms: [50.0, 400.0, 20.0]
```

Very broad grids can significantly increase runtime. Start with narrow, physically motivated ranges and expand them only if needed.
When to use each mode, in practice:

- Start with `independent` if you want the simplest robust disk fit.
- Use `disk_arctan` if you want a smooth parametric rotation curve.
- Use `disk_kepler`, `NSC`, or `Plummer` only if those physical assumptions match your science case.


## Outflow model

The outflow configuration is used when `component_mode` is `outflow` or `disk_then_outflow`. This section controls the cone geometry, shelling, and outflow velocity grid.

This is usually the most sensitive part of the model and should be adjusted carefully based on the observed morphology.

```yaml
fit:
  outflow:
    radius_range_arcsec: [0.0, 35.0]
    num_shells: 35
    pa_deg: 110
    opening_deg: 110
    double_cone: true
    mask_mode: bicone
    beta_grid_deg: [60, 130, 5]
    v_grid_kms: [0.0, 1500.0, 15.0]
```

Most important fields:

- `fit.outflow.radius_range_arcsec`
  Radial extent of the outflow fit.
- `fit.outflow.num_shells`
  Number of outflow shells.
- `fit.outflow.pa_deg`
  Outflow axis position angle.
- `fit.outflow.opening_deg`
  Cone opening angle.
- `fit.outflow.double_cone`
  Whether both lobes are modeled.
- `fit.outflow.mask_mode`
  `single` or `bicone`.
- `fit.outflow.beta_grid_deg`
  Inclination grid for the outflow.
- `fit.outflow.v_grid_kms`
  Outflow velocity grid.

Typical choices:

- Use `mask_mode: bicone` for a two-sided outflow.
- Use `mask_mode: single` if you only want one cone.
- Keep `opening_deg` in the physically plausible range for your target.
- Start with a narrower `v_grid_kms` than the example if you already know the expected outflow speed scale.

Single-cone example:

```yaml
fit:
  outflow:
    mask_mode: single
    double_cone: false
    pa_deg: 110
    opening_deg: 80
```

## Advanced options

Most users should leave this section unchanged for initial runs. Most users can leave this section close to the example config and only change a few fields. Change this section only after the basic run works.

```yaml
advanced:
  npt: 200000
  perc_disc: [0.05, 0.95]
  perc_out: [0.01, 0.99]
  compute_energetics: false
  compute_escape_fraction: false
```

Most important fields:

- `advanced.npt`
  Monte Carlo sample size. Larger values can improve smoothness but increase runtime.
- `advanced.perc_disc`, `advanced.perc_out`
  Percentiles used in the comparison between data and model.
- `advanced.use_crps`
  Switches the loss metric.
- `advanced.compute_energetics`
  Enables outflow energetics if the required inputs are available.
- `advanced.compute_escape_fraction`
  Enables the escape-velocity diagnostic.

Practical guidance:

- Leave `npt` unchanged for the first run.
- Reduce `npt` or narrow parameter grids if runtime is too long.
- Only enable `compute_energetics` once the basic fit is working and your line/unit inputs are correct.

## Output options

This section controls which user-facing products are written. Most users only need to decide whether they want plots, the summary JSON, and the larger optional FITS products.

```yaml
output:
  save_plots: true
  show_plots: false
  save_summary_json: true
  save_run_config_copy: true
  overwrite: true
```

Most important fields:

- `output.save_plots`
  Save diagnostic figures to disk.
- `output.show_plots`
  Display plots interactively. Usually keep this `false` for batch runs.
- `output.save_summary_json`
  Write `summary.json`.
- `output.save_run_config_copy`
  Save a copy of the YAML config used for the run.

Also note:

- `input.save_all_outputs: true`
  Enables additional outputs, including more FITS products.

For a minimal run with fewer files:

```yaml
input:
  save_all_outputs: false

output:
  save_plots: true
  save_summary_json: true
```

## Minimal working example

If you want the smallest realistic starting point for your own cube, this is a good template:

```yaml
paths:
  data_dir: /path/to/data
  ancillary_dir: /path/to/ancillary
  output_dir: /path/to/outputs

input:
  cube_file: my_cube.fits
  sn_map: null
  save_all_outputs: false

target:
  agn_ra: null
  agn_dec: null
  center_mode: flux
  center_xy_manual: null
  redshift: 0.005

line:
  wavelength_line: 5006.8
  wavelength_line_unit: Angstrom

processing:
  sn_thresh: 3
  nrebin: 1
  pixel_scale_arcsec_manual: null
  psf_sigma: 0.9
  lsf_sigma: 70.0
  vel_sigma: 0.0

fit:
  component_mode: disk
  disc:
    mode: independent
    radius_range_arcsec: [0.0, 20.0]
    num_shells: 15
    beta_grid_deg: [40, 90, 5]
    independent:
      v_grid_kms: [0.0, 300.0, 20.0]

advanced:
  npt: 200000
  compute_energetics: false

output:
  save_plots: true
  show_plots: false
  save_summary_json: true
```

Key lines to change first:

- `cube_file`
- `redshift`
- `wavelength_line` and `wavelength_line_unit`
- `component_mode`
- the disk or outflow parameter grids

## Common mistakes

If something goes wrong, check these before changing model parameters:


### Wrong working directory

If your config uses relative paths such as `./Data` or `./Ancillary_material`, running the command from the wrong directory will break otherwise-correct settings.

Fix:

```bash
cd /path/to/the/config_directory
moka3d validate config.yaml
```

### Wrong paths

If the cube or ancillary maps are not found, check:

- `paths.data_dir`
- `paths.ancillary_dir`
- `input.cube_file`
- `input.sn_map`
- `input.ne_map`

If you use relative paths, they are resolved from the directory where you run the command.

### Inconsistent line units

Make sure `line.wavelength_line` and `line.wavelength_line_unit` match the actual line you want to fit. For example, do not provide a frequency value with `Angstrom` as the unit.

### Invalid parameter grids

Grid fields such as:

- `beta_grid_deg`
- `v_grid_kms`
- `vmax_grid_kms`

must have the correct number of values and a positive step. If validation fails, check the shape of the list first.

### Forgetting to validate

Always validate before running:

```bash
moka3d validate config.yaml
```

This catches missing files, invalid choices, and malformed grids before the full run starts.

## Validation reminder

Before every real run, use:

```bash
moka3d validate config.yaml
```

If validation succeeds, you can then run:

```bash
moka3d run config.yaml
```
