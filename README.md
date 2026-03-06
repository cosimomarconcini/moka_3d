# moka_3d

MOKA3D is a Python package for **3D kinematic modeling of emission-line gas in galaxies**, designed to analyze integral-field spectroscopy data cubes.

The code reconstructs gas kinematics directly in the **data cube space**, comparing observed and modeled spectral profiles to constrain the velocity structure of ionized gas.

Typical applications include:

- AGN-driven outflows
- rotating gas discs
- combined disc + outflow systems

---

# Features

- Fit **rotating discs**
- Fit **biconical outflows**
- Combined **disc + outflow modeling**
- Direct comparison between model and observed data cubes
- Shell-based radial kinematic modeling
- Percentile-based velocity diagnostics
- Full diagnostic plots and residual maps
- Command-line interface for reproducible workflows

---
# Layout
moka3d/
├── src/moka3d/           
│   ├── cli.py
│   ├── config.py
│   ├── defaults.py
│   ├── pipeline.py
│   ├── plotting.py
│   ├── moka3d_source.py
│   └── rotation_curves.py
│
├── examples/             
│   └── config_basic.yaml
│
├── Data/                 Input FITS cubes
├── Ancillary_material/   SN maps or other auxiliary data
├── Outputs/              
# Installation

Clone the repository:

from terminal run:

git clone https://github.com/cosimomarconcini/moka_3d.git
cd moka3d

