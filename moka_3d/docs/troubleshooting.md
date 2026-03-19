# Troubleshooting

This page should collect the most common user-facing failure modes and the shortest reliable fixes. It should be practical and symptom-driven.

## Installation Problems

- Missing dependencies.
- CLI not found after install.
- Confusion about repository root versus package root.

This section should show the minimum commands needed to verify a working installation.

## Configuration Validation Errors

- Invalid `component_mode`.
- Missing required line settings.
- Invalid Doppler convention values.

This section should explain how to use `moka3d validate` and how to interpret validation failures.

## FITS and WCS Problems

- Pixel scale cannot be inferred.
- Spectral axis is malformed or unsupported.
- Center coordinates do not match the data.

This section should explain what metadata MOKA<sup>3D</sup> expects from input cubes and when a manual override is needed.

## Energetics Problems

- Unsupported emission line.
- Missing or malformed `BUNIT`.
- Missing density information.

This section should explain when energetics are skipped intentionally and what inputs are required to enable them.

## Runtime and Performance Problems

- Large grids run slowly.
- Many shells increase runtime.
- Plot-heavy runs are slower than fit-only runs.

This section should give practical advice for reducing runtime without promising internal performance guarantees.

## Reporting a Problem

- Include the config file.
- Include the command that was run.
- Include the traceback or warning message.
- Include package version and environment details.

This section should describe the minimum bug report needed for maintainers to reproduce an issue.
