# Understanding Outputs

MOKA3D writes results to the directory defined by `paths.output_dir` in your YAML configuration.
This is the first place to look after every run.

Each run creates its own output folder, so you can compare runs without overwriting earlier results.

## Output directory structure

After a successful run, you will usually see a structure like this:

```text
Outputs/
└── YYYY-MM-DD_HHMMSS_<cube_name>/
    ├── summary.json
    ├── moka3d.log
    ├── *.png
    └── *.fits
```

The timestamp ensures that runs are never overwritten.

The exact set of files depends on your configuration. In particular:

- `output_dir` controls the top-level destination
- `output.save_plots` controls whether plots are written
- `input.save_all_outputs` controls whether additional FITS products are written

Each run gets a separate timestamped folder. This is the directory you should inspect first after `moka3d run config.yaml` finishes.

## Key outputs

### `summary.json`

This is the fastest way to inspect a completed run. It contains the main run metadata and the best-fit quantities that most users want to check first.

Typical contents include:

- which component mode was used
- basic target and line metadata
- estimated center and orientation information
- best-fit disk and/or outflow parameters
- optional derived quantities when enabled

Use `summary.json` when you want to:

- quickly check whether the fit converged to plausible values
- compare two runs
- read results from a script or notebook

The exact structure may evolve between versions, so avoid relying on hard-coded keys unless you control the environment.

### Diagnostic plots

The plot files are your first visual sanity check. They are usually more informative than looking at a long list of parameter values in isolation.

Typical plots show:

- data versus model maps
- residual maps
- masks and shell geometry
- fit summaries or parameter-grid diagnostics

Use the plots to answer simple questions first:

- does the model reproduce the large-scale velocity pattern?
- are the residuals mostly small and unstructured?
- do the fitted masks and shells look aligned with the target?

These plots are usually the fastest way to decide whether a run is usable.

### Optional FITS products

If `input.save_all_outputs: true`, MOKA3D writes additional FITS files. These are useful when you want to inspect results in detail or reuse them in your own analysis.

At a high level, these products can include:

- model cubes
- residual cubes
- moment-like maps such as flux, velocity, and velocity dispersion
- optional shell-wise or derived products when enabled

These files can be large, so they are disabled by default for faster runs.

You do not need every FITS file for a first check. In most cases, start with `summary.json`, then the plots, then open the FITS products only if you need a closer look.

## How to interpret results

A good fit usually has these practical signs:

- the model velocity field follows the same large-scale pattern as the data
- residual maps do not show strong coherent structures over large regions
- the best-fit parameters are physically plausible for the target

Large residuals do not always mean the run failed. They often mean one of these is true:

- the parameter grid is too narrow or too coarse
- the selected fit mode is not appropriate for the source
- the masks or center are not well matched to the data
- the source contains structure that the chosen model does not capture

In many cases, they highlight where the model assumptions are too simple for the data.

Common red flags:

- strong structured residuals rather than weak patchy residuals
- clear mismatch between model and data in the velocity field
- best-fit values at the edge of the tested grid
- physically implausible disk or outflow parameters

If you see these, do not immediately trust the numbers in `summary.json`. Adjust the configuration and rerun.

## Typical workflow after a run

For most real analyses, a sensible sequence is:

1. Open `summary.json` and check the main fitted parameters.
2. Look at the diagnostic plots and decide whether the fit is qualitatively reasonable.
3. If needed, adjust the configuration:
   - change `fit.component_mode`
   - narrow or widen parameter grids
   - improve masks or ancillary inputs
   - review line, redshift, or PSF/LSF settings
4. Run `moka3d validate config.yaml` again.
5. Rerun the pipeline.

In practice, most users iterate a few times before settling on a production run.

## Minimal examples

### Read `summary.json` in Python

```python
import json
from pathlib import Path

summary_path = Path("Outputs/<run_name>/summary.json")
summary = json.loads(summary_path.read_text())

print(summary.keys())
print(summary)
```

### Inspect a FITS output

```python
from astropy.io import fits

fits_path = "Outputs/<run_name>/bestfit_moment_maps.fits"
with fits.open(fits_path) as hdul:
    print(hdul.info())
```

## What to check first

If you want the shortest practical checklist after a run:

1. confirm that `summary.json` exists and contains reasonable values
2. confirm that `moka3d.log` does not contain obvious failures
3. inspect the main diagnostic plots
4. open optional FITS products only if the summary and plots look reasonable

## Next steps

- Configuration Guide (configuration.md)

Use the configuration guide when you need to change model setup or parameter grids.
