# Quickstart

This guide walks you from a fresh clone of the repository to a successful end-to-end run of MOKA<sup>3D</sup> using the bundled example data.

## What you will do

- Clone the repository and install the `moka3d` command-line tool.
- Run the bundled example configuration on the sample FITS data included in the repository.
- Check the output directory and confirm that the run completed correctly.

## Requirements

- Python 3.9 or newer
- `git` installed and available in your terminal
- Optional: `conda` if you prefer working in an isolated environment

If you want to use a conda environment, the simplest option is:

```bash
conda env create -f environment.yml
conda activate moka3d
```

If you prefer to create the environment manually, use:

```bash
conda create -n moka3d_env python=3.11
conda activate moka3d_env
```

## Step 1 — Get the code

Clone the repository and move into the repository root:

```bash
git clone https://github.com/cosimomarconcini/moka3d.git
cd moka3d
```

At this point you are in the top-level repository directory. Inside it, there is a subdirectory also named `moka3d/` that contains the installable package, the example configuration, and the bundled sample data.

## Step 2 — Install the package

Install the package from the repository root:

```bash
python -m pip install -e ./moka3d
```

If you created the environment with `conda env create -f environment.yml`, this editable install step is already included.

Check that the command-line interface is available:

```bash
moka3d --help
```

If this prints the CLI help message, the installation worked.

## Step 3 — Move to the example directory

The bundled example configuration uses relative paths such as `./Data` and `./Ancillary_material`. For that reason, you must run the example from inside the repository subdirectory `moka3d/`, not from the repository root.

Move into the example directory:

```bash
cd moka3d
```

After this step, you should be inside the directory that contains:

- `config_moka.yaml`
- `Data/`
- `Ancillary_material/`

## Step 4 — Run the example

Validate the bundled example configuration:

```bash
moka3d validate config_moka.yaml
```

Run the bundled example:

```bash
moka3d run config_moka.yaml
```

(Optional) Write a template configuration file for your own future runs:

```bash
moka3d init-config example_config.yaml
```

The example uses:

- `config_moka.yaml`
- sample FITS cubes in `Data/`
- ancillary maps in `Ancillary_material/`

## Step 5 — Check results

When the run completes, a new timestamped directory will be created under:

```text
Outputs/YYYY-MM-DD_HHMMSS_<cube_name>/
```

List the output directories:

```bash
ls Outputs
```

Inspect the newest run directory:

```bash
ls Outputs/<timestamped_run_directory>
```

If the run completed successfully, you should see files such as `summary.json` and multiple `.png` plots.

The run directory typically contains:

- `summary.json`  
  Machine-readable summary of the run configuration and best-fit results.
- `moka3d.log`  
  Execution log with warnings, progress, and runtime details.
- plot files (`.png`)  
  Diagnostic figures such as residual maps, shell overlays, and map comparisons.
- optional FITS outputs  
  Best-fit model cubes, moment maps, and derived tables, depending on configuration.

## Common pitfalls

If something does not work, check the following common issues:

### Running from the wrong directory

If you run `moka3d run config_moka.yaml` from the repository root instead of from `moka3d/`, the example may fail because the config uses relative paths like `./Data`.

Fix:

```bash
cd moka3d
```

### Missing dependencies

If `moka3d --help` fails after installation, your environment is incomplete or the install did not finish correctly.

Fix:

```bash
python -m pip install -e ./moka3d
moka3d --help
```

### Configuration validation errors

If `moka3d validate config_moka.yaml` reports an error, do not run the pipeline yet. Fix the configuration first, then validate again.

Use:

```bash
moka3d validate config_moka.yaml
```

until it succeeds.

## Next steps

- [Configuration Guide](configuration.md)
- [Outputs Guide](outputs.md)

Use the configuration guide to adapt the YAML file to your own data, and the outputs guide to interpret the results of a completed run.
