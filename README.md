# moka_3d

MOKA3D is a Python package for **3D kinematic modeling of emission-line gas in galaxies**, designed to analyze integral-field spectroscopy data cubes.

The code reconstructs gas kinematics directly in the **data cube space**, comparing observed and modeled spectral profiles to constrain the velocity structure of ionized gas.

Typical applications include:

- AGN/SF-driven outflows
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
        
# Installation
I suggest to create a new environment and then install MOKA 3D as explained in the following.
E.g.:
conda create --name moka3d_env
Then activate it:
conda activate moka3d_env

Then, clone the MOKA 3D repository. For now this repo is private to to clone it you will need a personal token, you can do it like this:
Settings--> Developer settings --> Personal access tokens --> Tokens --> Generate new token 

Then, from terminal run:

git clone https://github.com/cosimomarconcini/moka_3d.git

And fill this:

Username: your GitHub username (not email, username)
Password: the token

cd moka3d

