# moka_3d

MOKA3D is a Python code for 3D kinematic modeling of emission-line gas in galaxies, designed to analyze integral-field spectroscopy data cubes.

The code reconstructs gas kinematics directly in the data cube space, comparing observed and modeled spectral profiles to constrain the velocity structure of ionized gas.

Typical applications include:

- AGN/SF-driven outflows
- rotating gas discs
- combined disc + outflow systems

---

# Features

- Fit rotating discs
- Fit biconical outflows
- Combined disc + outflow modeling
- Direct comparison between model and observed data cubes
- Shell-based radial kinematic modeling
- Percentile-based velocity diagnostics
- Full diagnostic plots and residual maps
- Command-line interface for reproducible workflows

---
        
# Installation
I suggest to create a new environment from the environment_moka.yml file in this repo and then install MOKA 3D as explained in the following.
1) conda env create -f environment_moka.yml
2) conda activate moka3d_env



Then, clone the MOKA 3D repository. For now this repo is private to to clone it you will need a personal token, you can do it like this:

Click your profile picture at the top right --> Settings --> Scroll down the left panel --> Developer settings --> Personal access tokens --> Tokens (classic) --> Generate new token (classic) --> select only 'repo' --> give it a random name --> scroll down and create token --> save token 

Then, from terminal run:

git clone https://github.com/cosimomarconcini/moka_3d.git

And fill this:

Username: your GitHub username (not email, username)
Password: the token

# Run MOKA 3D

Move to the pulled folder:

cd moka_3d/moka3d

Install the moka3d package:

python -m pip install -e .

Then test moka3d. This will run the config_basic.yaml file with all the necessary info on your data and fit type.

moka3d run config_basic.yaml 

# Citing MOKA 3D in your work

If you use this code in your publication please cite the following paper: https://ui.adsabs.harvard.edu/abs/2023A%26A...677A..58M/abstract 





