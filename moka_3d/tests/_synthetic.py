from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


C_KMS = 299792.458
DEFAULT_SHAPE_3D = (21, 13, 13)
DEFAULT_CENTER_XY = (6, 6)
DEFAULT_PIXEL_SCALE_ARCSEC = 0.2
DEFAULT_REDSHIFT = 0.01
DEFAULT_WL_LINE_ANGSTROM = 5006.8
DEFAULT_FREQ_LINE_HZ = 345.796e9
DEFAULT_BUNIT = "10**(-20) erg s-1 cm-2 Angstrom-1"


@dataclass(frozen=True)
class SyntheticCubeCase:
    name: str
    cube_path: Path
    data_dir: Path
    ancillary_dir: Path
    shape: tuple[int, int, int]
    center_xy: tuple[int, int]
    spectral_kind: str
    line_value: float
    line_unit: str
    redshift: float
    sn_map_path: Path | None = None
    ne_map_path: Path | None = None
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def cube_file(self) -> str:
        return self.cube_path.name

    @property
    def sn_map_file(self) -> str | None:
        return None if self.sn_map_path is None else self.sn_map_path.name

    @property
    def ne_map_file(self) -> str | None:
        return None if self.ne_map_path is None else self.ne_map_path.name


def ensure_workspace(tmp_path: Path) -> dict[str, Path]:
    data_dir = tmp_path / "Data"
    ancillary_dir = tmp_path / "Ancillary_material"
    data_dir.mkdir(parents=True, exist_ok=True)
    ancillary_dir.mkdir(parents=True, exist_ok=True)
    return {"root": tmp_path, "data_dir": data_dir, "ancillary_dir": ancillary_dir}


def velocity_axis_kms(
    n_spec: int = DEFAULT_SHAPE_3D[0],
    vmin_kms: float = -300.0,
    vmax_kms: float = 300.0,
) -> np.ndarray:
    return np.linspace(vmin_kms, vmax_kms, n_spec, dtype=float)


def wavelength_axis_angstrom(
    v_axis_kms: np.ndarray,
    rest_wavelength_angstrom: float = DEFAULT_WL_LINE_ANGSTROM,
    redshift: float = DEFAULT_REDSHIFT,
) -> np.ndarray:
    obs_wavelength = float(rest_wavelength_angstrom) * (1.0 + float(redshift))
    return obs_wavelength * (1.0 + np.asarray(v_axis_kms, dtype=float) / C_KMS)


def frequency_axis_hz(
    v_axis_kms: np.ndarray,
    rest_frequency_hz: float = DEFAULT_FREQ_LINE_HZ,
    redshift: float = DEFAULT_REDSHIFT,
) -> np.ndarray:
    obs_frequency = float(rest_frequency_hz) / (1.0 + float(redshift))
    return obs_frequency * (1.0 - np.asarray(v_axis_kms, dtype=float) / C_KMS)


def cube_from_vfield(
    v_axis_kms: np.ndarray,
    amp_map: np.ndarray,
    vlos_map: np.ndarray,
    sigma_kms: float,
) -> np.ndarray:
    dv = np.asarray(v_axis_kms, dtype=float)[:, None, None] - np.asarray(vlos_map, dtype=float)[None, :, :]
    cube = np.asarray(amp_map, dtype=float)[None, :, :] * np.exp(-0.5 * (dv / float(sigma_kms)) ** 2)
    return cube.astype(np.float32, copy=False)


def common_spatial_header(
    shape_yx: tuple[int, int] = DEFAULT_SHAPE_3D[1:],
    *,
    pixel_scale_arcsec: float = DEFAULT_PIXEL_SCALE_ARCSEC,
    ra_deg: float = 150.0,
    dec_deg: float = 2.0,
) -> fits.Header:
    ny, nx = shape_yx
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = int(nx)
    hdr["NAXIS2"] = int(ny)
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CRVAL1"] = float(ra_deg)
    hdr["CRVAL2"] = float(dec_deg)
    hdr["CRPIX1"] = float((nx + 1) / 2.0)
    hdr["CRPIX2"] = float((ny + 1) / 2.0)
    hdr["CDELT1"] = -float(pixel_scale_arcsec) / 3600.0
    hdr["CDELT2"] = float(pixel_scale_arcsec) / 3600.0
    return hdr


def build_cube_header(
    spectral_axis: np.ndarray,
    *,
    shape_yx: tuple[int, int] = DEFAULT_SHAPE_3D[1:],
    ctype3: str,
    cunit3: str,
    bunit: str | None = None,
) -> fits.Header:
    spectral_axis = np.asarray(spectral_axis, dtype=float)
    if spectral_axis.ndim != 1 or spectral_axis.size < 2:
        raise ValueError("spectral_axis must be a 1D array with at least two elements.")

    delta = np.diff(spectral_axis)
    linearity_atol = max(1e-9, abs(delta[0]) * 1e-9)
    if not np.allclose(delta, delta[0], rtol=0.0, atol=linearity_atol):
        raise ValueError("Synthetic spectral axis must be linear.")

    hdr = common_spatial_header(shape_yx=shape_yx)
    hdr["NAXIS3"] = int(spectral_axis.size)
    hdr["CTYPE3"] = ctype3
    hdr["CUNIT3"] = cunit3
    hdr["CRPIX3"] = 1.0
    hdr["CRVAL3"] = float(spectral_axis[0])
    hdr["CDELT3"] = float(delta[0])
    if bunit is not None:
        hdr["BUNIT"] = str(bunit)
    return hdr


def write_fits_cube(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)
    return path


def write_fits_image(path: Path, data: np.ndarray, header: fits.Header | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)
    return path


def make_sn_map_allgood(
    shape_yx: tuple[int, int] = DEFAULT_SHAPE_3D[1:],
    *,
    sn_value: float = 50.0,
) -> np.ndarray:
    return np.full(shape_yx, float(sn_value), dtype=np.float32)


def make_ne_map_step(
    shape_yx: tuple[int, int] = DEFAULT_SHAPE_3D[1:],
    *,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    inner_radius_pix: float = 2.0,
    inner_ne_cm3: float = 500.0,
    outer_ne_cm3: float = 200.0,
) -> np.ndarray:
    _, _, rr = centered_coordinates(shape_yx, center_xy)
    ne_map = np.full(shape_yx, float(outer_ne_cm3), dtype=np.float32)
    ne_map[rr <= float(inner_radius_pix)] = float(inner_ne_cm3)
    return ne_map


def centered_coordinates(
    shape_yx: tuple[int, int],
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape_yx, dtype=float)
    x = xx - float(center_xy[0])
    y = yy - float(center_xy[1])
    rr = np.hypot(x, y)
    return x, y, rr


def rotate_by_pa(
    x: np.ndarray,
    y: np.ndarray,
    pa_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    pa_rad = np.deg2rad(float(pa_deg))
    major_x = np.sin(pa_rad)
    major_y = np.cos(pa_rad)
    minor_x = np.cos(pa_rad)
    minor_y = -np.sin(pa_rad)
    x_rot = x * major_x + y * major_y
    y_rot = x * minor_x + y * minor_y
    return x_rot, y_rot


def make_disk_maps(
    shape_yx: tuple[int, int] = DEFAULT_SHAPE_3D[1:],
    *,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    pa_deg: float = 35.0,
    inc_deg: float = 60.0,
    vmax_kms: float = 140.0,
    r_turn_pix: float = 2.0,
    flux_scale_pix: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, _ = centered_coordinates(shape_yx, center_xy)
    x_rot, y_rot = rotate_by_pa(x, y, pa_deg)
    cosi = max(np.cos(np.deg2rad(float(inc_deg))), 0.2)
    r_disk = np.sqrt(x_rot**2 + (y_rot / cosi) ** 2)
    amp_map = np.exp(-r_disk / float(flux_scale_pix))
    vlos_map = float(vmax_kms) * np.sin(np.deg2rad(float(inc_deg))) * x_rot / np.sqrt(x_rot**2 + float(r_turn_pix) ** 2)
    return amp_map.astype(np.float32), vlos_map.astype(np.float32)


def make_bicone_maps(
    shape_yx: tuple[int, int] = DEFAULT_SHAPE_3D[1:],
    *,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    pa_deg: float = 90.0,
    opening_deg: float = 100.0,
    v_lobe_kms: float = 220.0,
    flux_scale_pix: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, rr = centered_coordinates(shape_yx, center_xy)
    pa_rad = np.deg2rad(float(pa_deg))
    ax = np.sin(pa_rad)
    ay = np.cos(pa_rad)

    dot = x * ax + y * ay
    safe_rr = np.where(rr == 0.0, 1.0, rr)
    cosang = dot / safe_rr
    inside = np.abs(cosang) >= np.cos(np.deg2rad(float(opening_deg)) / 2.0)
    inside[rr == 0.0] = False

    amp_map = 0.6 * np.exp(-rr / float(flux_scale_pix)) * inside
    sign = np.where(dot >= 0.0, 1.0, -1.0)
    vlos_map = float(v_lobe_kms) * sign * inside
    return amp_map.astype(np.float32), vlos_map.astype(np.float32)


def make_disk_wavelength_case(
    data_dir: Path,
    ancillary_dir: Path,
    *,
    name: str = "fx_disk_wl_small",
    v_axis_kms_values: np.ndarray | None = None,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    redshift: float = DEFAULT_REDSHIFT,
) -> SyntheticCubeCase:
    v_axis = velocity_axis_kms() if v_axis_kms_values is None else np.asarray(v_axis_kms_values, dtype=float)
    amp_map, vlos_map = make_disk_maps(center_xy=center_xy)
    cube = cube_from_vfield(v_axis, amp_map, vlos_map, sigma_kms=30.0)
    spectral_axis = wavelength_axis_angstrom(v_axis, redshift=redshift)
    header = build_cube_header(spectral_axis, ctype3="WAVE", cunit3="Angstrom", bunit=DEFAULT_BUNIT)
    cube_path = write_fits_cube(data_dir / f"{name}.fits", cube, header)

    return SyntheticCubeCase(
        name=name,
        cube_path=cube_path,
        data_dir=data_dir,
        ancillary_dir=ancillary_dir,
        shape=cube.shape,
        center_xy=center_xy,
        spectral_kind="wavelength",
        line_value=DEFAULT_WL_LINE_ANGSTROM,
        line_unit="Angstrom",
        redshift=redshift,
        expected={
            "velocity_axis_kms": v_axis,
            "disk_pa_deg": 35.0,
            "disk_inc_deg": 60.0,
            "disk_vmax_kms": 140.0,
            "sigma_kms": 30.0,
        },
    )


def make_outflow_wavelength_case(
    data_dir: Path,
    ancillary_dir: Path,
    *,
    name: str = "fx_outflow_wl_bicone_small",
    v_axis_kms_values: np.ndarray | None = None,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    redshift: float = DEFAULT_REDSHIFT,
) -> SyntheticCubeCase:
    v_axis = velocity_axis_kms() if v_axis_kms_values is None else np.asarray(v_axis_kms_values, dtype=float)
    amp_map, vlos_map = make_bicone_maps(center_xy=center_xy)
    cube = cube_from_vfield(v_axis, amp_map, vlos_map, sigma_kms=35.0)
    spectral_axis = wavelength_axis_angstrom(v_axis, redshift=redshift)
    header = build_cube_header(spectral_axis, ctype3="WAVE", cunit3="Angstrom", bunit=DEFAULT_BUNIT)
    cube_path = write_fits_cube(data_dir / f"{name}.fits", cube, header)

    return SyntheticCubeCase(
        name=name,
        cube_path=cube_path,
        data_dir=data_dir,
        ancillary_dir=ancillary_dir,
        shape=cube.shape,
        center_xy=center_xy,
        spectral_kind="wavelength",
        line_value=DEFAULT_WL_LINE_ANGSTROM,
        line_unit="Angstrom",
        redshift=redshift,
        expected={
            "velocity_axis_kms": v_axis,
            "outflow_pa_deg": 90.0,
            "outflow_opening_deg": 100.0,
            "outflow_v_kms": 220.0,
            "sigma_kms": 35.0,
        },
    )


def make_combo_wavelength_case(
    data_dir: Path,
    ancillary_dir: Path,
    *,
    name: str = "fx_combo_wl_small",
    v_axis_kms_values: np.ndarray | None = None,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    redshift: float = DEFAULT_REDSHIFT,
) -> SyntheticCubeCase:
    v_axis = velocity_axis_kms() if v_axis_kms_values is None else np.asarray(v_axis_kms_values, dtype=float)
    disk_amp, disk_vlos = make_disk_maps(center_xy=center_xy)
    out_amp, out_vlos = make_bicone_maps(center_xy=center_xy)
    disk_cube = cube_from_vfield(v_axis, disk_amp, disk_vlos, sigma_kms=30.0)
    out_cube = cube_from_vfield(v_axis, out_amp, out_vlos, sigma_kms=35.0)
    cube = (disk_cube + 0.7 * out_cube).astype(np.float32)
    spectral_axis = wavelength_axis_angstrom(v_axis, redshift=redshift)
    header = build_cube_header(spectral_axis, ctype3="WAVE", cunit3="Angstrom", bunit=DEFAULT_BUNIT)
    cube_path = write_fits_cube(data_dir / f"{name}.fits", cube, header)

    return SyntheticCubeCase(
        name=name,
        cube_path=cube_path,
        data_dir=data_dir,
        ancillary_dir=ancillary_dir,
        shape=cube.shape,
        center_xy=center_xy,
        spectral_kind="wavelength",
        line_value=DEFAULT_WL_LINE_ANGSTROM,
        line_unit="Angstrom",
        redshift=redshift,
        expected={
            "velocity_axis_kms": v_axis,
            "disk_pa_deg": 35.0,
            "disk_inc_deg": 60.0,
            "disk_vmax_kms": 140.0,
            "outflow_pa_deg": 90.0,
            "outflow_opening_deg": 100.0,
            "outflow_v_kms": 220.0,
        },
    )


def make_disk_frequency_case(
    data_dir: Path,
    ancillary_dir: Path,
    *,
    name: str = "fx_disk_freq_small",
    v_axis_kms_values: np.ndarray | None = None,
    center_xy: tuple[int, int] = DEFAULT_CENTER_XY,
    redshift: float = DEFAULT_REDSHIFT,
) -> SyntheticCubeCase:
    v_axis = velocity_axis_kms() if v_axis_kms_values is None else np.asarray(v_axis_kms_values, dtype=float)
    amp_map, vlos_map = make_disk_maps(center_xy=center_xy)
    cube = cube_from_vfield(v_axis, amp_map, vlos_map, sigma_kms=30.0)
    spectral_axis = frequency_axis_hz(v_axis, redshift=redshift)
    header = build_cube_header(spectral_axis, ctype3="FREQ", cunit3="Hz")
    cube_path = write_fits_cube(data_dir / f"{name}.fits", cube, header)

    return SyntheticCubeCase(
        name=name,
        cube_path=cube_path,
        data_dir=data_dir,
        ancillary_dir=ancillary_dir,
        shape=cube.shape,
        center_xy=center_xy,
        spectral_kind="frequency",
        line_value=DEFAULT_FREQ_LINE_HZ,
        line_unit="Hz",
        redshift=redshift,
        expected={
            "velocity_axis_kms": v_axis,
            "disk_pa_deg": 35.0,
            "disk_inc_deg": 60.0,
            "disk_vmax_kms": 140.0,
            "sigma_kms": 30.0,
        },
    )
