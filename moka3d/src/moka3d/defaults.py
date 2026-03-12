#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:38:07 2026

@author: cosimo
"""

DEFAULT_CONFIG_YAML = """\
paths:
  data_dir: ./Data
  ancillary_dir: ./Ancillary_material
  output_dir: ./Outputs

input:
  cube_file: NGC7582_test_OIII_smooth.fits
  sn_map: SN_map_NGC7582.fits
  save_all_outputs: false
  ne_map: null
  ne_outflow: null


target:
  agn_ra: '23h18m23.6280s'
  agn_dec: '-42d22m13.512s'
  center_mode: null
  center_xy_manual: null
  redshift: 0.005410

line:
  wavelength_line: 5006.8
  wavelength_line_unit: Angstrom

processing:
  sn_thresh: 3
  nrebin: 1
  xrange: null
  yrange: null
  pixel_scale_arcsec_manual: null
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
  psf_sigma: 1.0
  lsf_sigma: 72.0
  vel_sigma: 0.0
  logradius: false

maps:
  fluxmap: null
  velmap: null
  sigmap: null

fit:
  component_mode: disk_then_outflow

  disc:
    mode: independent

    radius_range_arcsec: [0.0, 40.0]
    num_shells: 30

    pa_deg: null
    pa_unc_deg: null

    beta_grid_deg: [50, 100, 5]

    independent:
      v_grid_kms: [0.0, 400.0, 20.0]

    kepler:
      mbh_grid_msun: [1.0e6, 1.0e11]
      n_geom: 50

    nsc:
      re_pc: 5.0
      a_grid: [1.0e-3, 1.0e3]
      n_geom: 50

    plummer:
      a_pc: 4.0
      m0_grid_msun: [1.0e6, 1.0e11]
      n_geom: 50

    arctan:
      rt_arcsec: null
      vmax_grid_kms: [0.0, 400.0, 20.0]

  outflow:
    radius_range_arcsec: [0.0, 30.0]
    num_shells: 30

    pa_deg: 105.0
    opening_deg: 120.0
    double_cone: true
    mask_mode: bicone

    beta_grid_deg: [50, 130, 5]
    v_grid_kms: [100.0, 1300.0, 20.0]


advanced:
  check_masking_before_fitting: false
  use_crps: false
  perc_disc: [0.05, 0.95]
  perc_out: [0.01, 0.99]
  perc_weights: [1, 1]
  npt: 200000
  use_global_beta_disc: true
  disc_theta_range:
    - [0, 1]
  disc_phi_range:
    - [0, 360]
  disc_zeta_range_mode: auto_from_psf
  use_global_beta_out: true
  mask_disk_with_outflow: true
  do_final_combined_model_plot: true
  outflow_axis_sign: 1
  resid_ranges: [0.15, 55, 55]
  compute_escape_fraction: false
  save_escape_fraction_table: true
  compute_energetics: false
  save_energetics_table: true
  assumed_ne_values: [100.0, 500.0, 1000.0]
  oiii_metallicity_z_over_zsun: 1.0


output:
  save_plots: true
  show_plots: false
  save_summary_json: true
  save_run_config_copy: true
  overwrite: true
"""