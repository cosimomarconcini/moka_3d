#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 4 15:31:00 2026

@author: cm
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import radians as rad
from astropy.io import fits
from . import rotations as rot
from . import amgraphics as amg
import fast_histogram as fh
from matplotlib import colors
from . import rotation_curves as rc
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from astropy import constants as const
from matplotlib.patches import Rectangle
import math
import os
import re
import sys
import astropy.units as u

from scipy.ndimage import gaussian_filter
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.constants import c
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Circle
from astropy.table import Table

from matplotlib.ticker import AutoMinorLocator
import json

from .plotting import finalize_figure
import logging
logger = logging.getLogger(__name__)

ACTION_LEVEL = 25
logging.addLevelName(ACTION_LEVEL, "ACTION")

def action(self, message, *args, **kwargs):
    if self.isEnabledFor(ACTION_LEVEL):
        self._log(ACTION_LEVEL, message, args, **kwargs)

logging.Logger.action = action


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[0m",
        "ACTION": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "moka3d.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter_file = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    formatter_console = ColorFormatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter_console)
    logger.addHandler(sh)




class utils():

    def __init__(self):
        
        pass
    
    
    def findrange(self, x):    
        """find step and range in axis """
        
        dx0 = x[1] - x[0]
        if not np.isfinite(dx0):
            finite_dx = (x[1:] - x[:-1])[np.isfinite(x[1:] - x[:-1])]
            if finite_dx.size == 0:
                raise ValueError("Axis spacing is all NaN/non-finite; cannot determine range.")
            dx0 = finite_dx[0]

        if np.isclose(dx0, 0.025):
            decimal = 3
        elif np.isclose(dx0, 39.0):
            decimal = 0
        else:
            decimal = 2



        xstep = np.unique(np.round(x[1:]-x[:-1], decimals=decimal))
        if xstep.size !=1:
            sys.exit('step non uniform')
        xrange = np.array([x[0]-xstep/2., x[-1]+xstep/2.])  # range limits are set at the external borders the two edges bin
                
        return xrange.flatten()
    
    def find_nearest(self, array, value): 
        """
        just a function to find the index of an element 
        in array closest to value, used in chan_maps
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def hist_edges(self, ranges, nbins):
        """
        Build the histogram edges
        """
        edges = [ np.linspace(ranges[0][0], ranges[0][1], nbins[0]+1), 
                  np.linspace(ranges[1][0], ranges[1][1], nbins[1]+1),
                  np.linspace(ranges[2][0], ranges[2][1], nbins[2]+1) ]
        return edges
        
    def write_cube(self, cube, filename=None): 
        """
        write model cube to fits file

        """
        
        
        data = cube['data']

        if filename is None:
            filename = 'cube.fits'
        hdu = fits.PrimaryHDU(data)
        hdu.header['CD1_1'] = cube['dx']
        hdu.header['CRPIX1'] = 1
        hdu.header['CRVAL1'] = cube['x'][0]
        hdu.header['CD2_2'] = cube['dy']
        hdu.header['CRPIX2'] = 1
        hdu.header['CRVAL2'] = cube['y'][0]   
        hdu.header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom' 
        if 'dv' in cube:
            hdu.header['CD3_3'] = cube['dv']
            hdu.header['CRPIX3'] = 1
            hdu.header['CRVAL3'] = cube['v'][0]
        elif 'dz' in cube:
            hdu.header['CD3_3'] = cube['dz']
            hdu.header['CRPIX3'] = 1
            hdu.header['CRVAL3'] = cube['z'][0]        
        
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)

    def kin_maps(self, fluxthr=1e-10, domap=None):
        """
        Create moment maps from model cube and save them
        in model dictionary, to be able to reuse them.
        """

        data = np.vstack([self.yobs_psf, self.xobs_psf]).T  # Y, X

        map0 = fh.histogramdd(
            data,
            bins=self.cube["nbins"][1:],
            range=self.cube["range"][1:],
            weights=self.flux,
        )

        map1 = fh.histogramdd(
            data,
            bins=self.cube["nbins"][1:],
            range=self.cube["range"][1:],
            weights=self.flux * self.vlos_lsf,
        )

        kmap = {}
        kmap["flux"] = np.array(map0, dtype=float, copy=True)

        # Pixels where flux is invalid or too low
        flux_max = np.nanmax(kmap["flux"])
        w = (~np.isfinite(kmap["flux"])) | (kmap["flux"] <= 0)

        if np.isfinite(flux_max) and flux_max > 0:
            w |= (kmap["flux"] < fluxthr * flux_max)

        kmap["flux"][w] = np.nan

        need_vel = domap in ("all", "vel", "sig")
        vel = None

        if need_vel:
            with np.errstate(divide="ignore", invalid="ignore"):
                vel = np.divide(
                    map1,
                    map0,
                    out=np.full_like(map1, np.nan, dtype=float),
                    where=(~w) & np.isfinite(map0) & (map0 > 0),
                )

            vel[w] = np.nan

            if domap in ("all", "vel"):
                kmap["vel"] = vel

        if domap in ("all", "sig"):
            map2 = fh.histogramdd(
                data,
                bins=self.cube["nbins"][1:],
                range=self.cube["range"][1:],
                weights=self.flux * self.vlos_lsf**2,
            )

            if vel is None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vel = np.divide(
                        map1,
                        map0,
                        out=np.full_like(map1, np.nan, dtype=float),
                        where=(~w) & np.isfinite(map0) & (map0 > 0),
                    )
                vel[w] = np.nan

            with np.errstate(divide="ignore", invalid="ignore"):
                second_moment = np.divide(
                    map2,
                    map0,
                    out=np.full_like(map2, np.nan, dtype=float),
                    where=(~w) & np.isfinite(map0) & (map0 > 0),
                )

            var = second_moment - vel**2
            var[~np.isfinite(var)] = np.nan
            var[var < 0] = np.nan

            sig = np.sqrt(var)
            sig[w] = np.nan
            kmap["sig"] = sig

        self.maps = kmap
        
            
    def kin_maps_cube(
            self,
            fluxthr=1e-10,
            flux=False,
            velocity=False,
            vel_disp=False,
            vel_array=None,
            new_method_mom_maps=False
        ):
        """
        Create mom0/mom1/mom2 maps from a cube.
    
        IMPORTANT FIXES:
          - DO NOT multiply flux by dv when it is used as the denominator for mom1/mom2.
            (Doing so shrinks velocities by ~1/dv.)
          - Use non-negative weights (Ipos) for moment calculations to avoid NaNs in sigma
            from negative numerator inside sqrt.
        """
    
        cube = self.cube
    
        # velocity axis for each channel (km/s) -> shape (nv,)
        # If you ever want to override with vel_array, you can uncomment below:
        # v = cube["v"] if vel_array is None else np.asarray(vel_array, dtype=float)
        v = np.asarray(cube["v"], dtype=float)
    
        # Build v(x,y) cube: shape (nv, ny, nx)
        v3d = np.transpose(np.tile(v, (cube["nx"], cube["ny"], 1)), (2, 1, 0))
    
        # Data cube: shape (nv, ny, nx)
        I = np.asarray(cube["data"], dtype=float)
    
        # Use only non-negative weights to avoid negative second moment and NaNs in sigma
        Ipos = np.where(np.isfinite(I) & (I > 0), I, 0.0)
    
        # Helper: threshold mask based on peak flux
        def _mask_from_flux(flux2d):
            fmax = np.nanmax(flux2d)
            if not np.isfinite(fmax) or fmax <= 0:
                return ~np.isfinite(flux2d)  # everything bad if no valid flux
            return (flux2d < fluxthr * fmax) | (~np.isfinite(flux2d))
    
        # ------------------------------------------------------------
        # Stepwise mode (kept for compatibility with your flags)
        # Note: your flag logic is unusual; we preserve it.
        # ------------------------------------------------------------
        if (flux is False) and (velocity is True) and (vel_disp is True):
            # mom0 (NO dv factor here)
            self.maps["flux"] = np.nansum(Ipos, axis=0)
            w = _mask_from_flux(self.maps["flux"])
            self.maps["flux"][w] = np.nan
    
        if (flux is True) and (velocity is False) and (vel_disp is True):
            # mom1
            flux2d = np.asarray(self.maps["flux"], dtype=float)
            w = _mask_from_flux(flux2d)
    
            num1 = np.nansum(v3d * Ipos, axis=0)
            vel2d = np.full_like(flux2d, np.nan, dtype=float)
    
            good = np.isfinite(flux2d) & (flux2d > 0) & (~w)
            vel2d[good] = num1[good] / flux2d[good]
    
            self.maps["vel"] = vel2d
    
        if (flux is True) and (velocity is True) and (vel_disp is False):
            # mom2
            flux2d = np.asarray(self.maps["flux"], dtype=float)
            vel2d = np.asarray(self.maps["vel"], dtype=float)
            w = _mask_from_flux(flux2d)
    
            vel3d = vel2d[None, :, :]  # broadcast
            num2 = np.nansum(((v3d - vel3d) ** 2) * Ipos, axis=0)
    
            # guard tiny negative due to roundoff
            num2 = np.where(num2 < 0, 0.0, num2)
    
            sig2d = np.full_like(flux2d, np.nan, dtype=float)
            good = np.isfinite(flux2d) & (flux2d > 0) & np.isfinite(vel2d) & (~w)
            sig2d[good] = np.sqrt(num2[good] / flux2d[good])
    
            self.maps["sig"] = sig2d
    
        # ------------------------------------------------------------
        # Default mode: compute all three maps at once
        # ------------------------------------------------------------
        if (flux is False) and (velocity is False) and (vel_disp is False) and (new_method_mom_maps is False):
            kmap = {}
    
            # mom0 (NO dv factor here)
            flux2d = np.nansum(Ipos, axis=0)
            kmap["flux"] = flux2d
    
            w = _mask_from_flux(flux2d)
    
            # mom1
            num1 = np.nansum(v3d * Ipos, axis=0)
            vel2d = np.full_like(flux2d, np.nan, dtype=float)
            good = np.isfinite(flux2d) & (flux2d > 0) & (~w)
            vel2d[good] = num1[good] / flux2d[good]
            kmap["vel"] = vel2d
    
            # mom2
            vel3d = vel2d[None, :, :]
            num2 = np.nansum(((v3d - vel3d) ** 2) * Ipos, axis=0)
            num2 = np.where(num2 < 0, 0.0, num2)
    
            sig2d = np.full_like(flux2d, np.nan, dtype=float)
            good2 = good & np.isfinite(vel2d)
            sig2d[good2] = np.sqrt(num2[good2] / flux2d[good2])
            kmap["sig"] = sig2d
    
            self.maps = kmap
    
        return self.maps
        
        
    def plot_kin_maps(
        self,
        flrange=None, vrange=None, sigrange=None,
        xrange=None, yrange=None, extent=None,
        xy_AGN=None,
        obs_too=None, cut_obs=None, xcbar=None,
        mom0=None, mom1=None, mom2=None,
        mask_over_range_sigma=False,
        residual_map=False
    ):
        """
        Plot moment maps.
    
        Current behavior kept:
          - If called as you do now (no obs_too and no mom0/mom1/mom2), it makes a 1x3 plot.
          - vrange/sigrange/flrange still accepted.
    
        Change:
          - Flux map is ALWAYS displayed in log10 scale.
        """
        cube = self.cube
        maps = self.maps
    
        # -----------------------------
        # Reject legacy branches we no longer support in the slim version
        # (keeps signature for compatibility, but fails loudly if you ever use them)
        # -----------------------------
        if obs_too is not None:
            raise NotImplementedError("plot_kin_maps: obs_too comparison mode removed in slim version.")
        if (mom0 is not None) or (mom1 is not None) or (mom2 is not None):
            raise NotImplementedError("plot_kin_maps: external mom0/mom1/mom2 mode removed in slim version.")
        if residual_map:
            raise NotImplementedError("plot_kin_maps: residual_map mode removed in slim version.")
        if cut_obs is not None or xcbar is not None:
            raise NotImplementedError("plot_kin_maps: cut_obs/xcbar options removed in slim version.")
    
        # -----------------------------
        # Defaults for ranges / extents
        # -----------------------------
        if xrange is None:
            xrange = cube["xextent"]
        if yrange is None:
            yrange = cube["yextent"]
    
        if extent is None:
            x0, x1 = cube["xextent"]
            y0, y1 = cube["yextent"]
        
            # If AGN position provided → recenter coordinates
            if xy_AGN is not None:
                extent = [
                    x0 - xy_AGN[0],
                    x1 - xy_AGN[0],
                    y0 - xy_AGN[1],
                    y1 - xy_AGN[1],
                ]
            else:
                extent = [x0, x1, y0, y1]
        if extent == "pix":
            extent = None
        if (extent is not None) and (xy_AGN is not None):
            xrange = [xrange[0] - xy_AGN[0], xrange[1] - xy_AGN[0]]
            yrange = [yrange[0] - xy_AGN[1], yrange[1] - xy_AGN[1]]
    
        # -----------------------------
        # Build log-flux map safely (avoid log10(<=0))
        # -----------------------------
        flux = np.array(maps["flux"], dtype=float)
        # treat non-positive as invalid for log
        flux_log = np.full_like(flux, np.nan, dtype=float)
        pos = np.isfinite(flux) & (flux > 0)
        flux_log[pos] = np.log10(flux[pos])
    
        # default flrange in *log space*
        if flrange is None:
            if np.any(np.isfinite(flux_log)):
                flrange = [np.nanpercentile(flux_log, 1), np.nanpercentile(flux_log, 99)]
            else:
                flrange = [-1.0, 1.0]  # fallback
        else:
            # user passed linear flux limits -> convert to log space
            lo = float(flrange[0])
            hi = float(flrange[1])
            if lo <= 0 or hi <= 0:
                raise ValueError("flrange must be > 0 when using log-flux display.")
            flrange = [np.log10(lo), np.log10(hi)]
    
        if vrange is None:
            vrange = [np.nanpercentile(maps["vel"], 5), np.nanpercentile(maps["vel"], 95)]
        if sigrange is None:
            sigrange = [np.nanpercentile(maps["sig"], 5), np.nanpercentile(maps["sig"], 95)]
    
        # Optional mask based on sigma range (kept because it is cheap and sometimes useful)
        if mask_over_range_sigma:
            k2 = (maps["sig"] > sigrange[1]) | (maps["sig"] < sigrange[0]) | ~np.isfinite(maps["sig"])
            flux_log = flux_log.copy()
            vel = np.array(maps["vel"], float).copy()
            sig = np.array(maps["sig"], float).copy()
            flux_log[k2] = np.nan
            vel[k2] = np.nan
            sig[k2] = np.nan
        else:
            vel = maps["vel"]
            sig = maps["sig"]
    
        cmap_fl = plt.cm.inferno
        cmap_vel = plt.cm.bwr
        cmap_sig = plt.cm.jet
        cmap_fl.set_bad("white")
        cmap_vel.set_bad("white")
        cmap_sig.set_bad("white")
    
        # -----------------------------
        # Plot: 1x3
        # -----------------------------
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True,
                               subplot_kw={"aspect": "equal"}, figsize=(24, 6))
    
        p1 = ax[0].imshow(flux_log, interpolation="nearest",
                          vmin=flrange[0], vmax=flrange[1],
                          extent=extent, origin="lower", cmap=cmap_fl)
        cb1 = fig.colorbar(p1, ax=ax[0], pad=0)
        cb1.set_label(r'$\log_{10}$ Flux [A.U.]', fontsize=22)
    
        p2 = ax[1].imshow(vel, interpolation="nearest",
                          vmin=vrange[0], vmax=vrange[1],
                          extent=extent, origin="lower", cmap=cmap_vel)
        cb2 = fig.colorbar(p2, ax=ax[1], pad=0)
        cb2.set_label(r"[km s$^{-1}$]", fontsize=22)
    
        p3 = ax[2].imshow(sig, interpolation="nearest",
                          vmin=sigrange[0], vmax=sigrange[1],
                          extent=extent, origin="lower", cmap=cmap_sig)
        cb3 = fig.colorbar(p3, ax=ax[2], pad=0)
        cb3.set_label(r"[km s$^{-1}$]", fontsize=22)
    
        if xy_AGN is not None:
            for j in range(3):
                if extent is None:
                    ax[j].scatter(xy_AGN[0], xy_AGN[1], s=200, marker='*', color='black', zorder=6)
                else:
                    ax[j].scatter(0.0, 0.0, s=200, marker='*', color='black' , zorder=6)
        
        ax[0].set_xlim(xrange[0], xrange[1])
        ax[0].set_ylim(yrange[0], yrange[1])
    
        ax[0].set_title("Log (Flux)", weight="bold", fontsize=26)
        ax[1].set_title("L.O.S. Velocity", weight="bold", fontsize=26)
        ax[2].set_title("Vel. dispersion", weight="bold", fontsize=26)
    
        ax[0].set_xlabel(r'$\Delta X$ [arcsec]')
        ax[0].set_ylabel(r'$\Delta Y$ [arcsec]')
        ax[1].set_xlabel(r'$\Delta X$ [arcsec]')
        ax[2].set_xlabel(r'$\Delta X$ [arcsec]')

    
        #plt.show()
        return maps
    
    def chan_maps(self, flrange=None, intervals=None, obs=False, residual = False, only_obs = False, cut_obs = False):
        """
        This function create velocity channel images at different intervals
                # flrange = range of flux
                # intervals = velocity intervals you want to observe the emission
                # obs = If not None, observed data cube, to compare the observed and model emission in certain velocity channel(s)
        """

        
        if flrange is None:
            flrange = [-np.nanpercentile(self.maps['flux'],1), np.nanpercentile(self.maps['flux'],99)]
        
        if intervals is None: # automatically select velocity intervals if not provided
            intervals = [-1300, -1000,-500, 0, 500, 1000, 1300]
        
        
        for j in range(len(intervals)): # adjust intervals, in case they exceed the model ones
            if intervals[j]<min(self.vlos):
                intervals[j] = int(min(self.vlos))
            elif intervals[j] > max(self.vlos):
                intervals[j] = int(max(self.vlos))
                
        rem_elem = 0
        for s in range(len(intervals)-1):
            if (intervals[s-rem_elem] == intervals[s+1-rem_elem]):
                intervals.pop(s-rem_elem)
                rem_elem+=1

        
        if obs == True: 
            if (residual == True) and (only_obs == False):
                fig, axes = plt.subplots(3,len(intervals)-1, sharex=True, sharey = True, subplot_kw={'aspect': 'equal'}, figsize = (12,6))
            elif (residual == False) and (only_obs == False):
                fig, axes = plt.subplots(2,len(intervals)-1, sharex=True, sharey = True, subplot_kw={'aspect': 'equal'}, figsize = (12,6))
            elif(residual == False) and (only_obs == True):
                fig, axes = plt.subplots(1,len(intervals)-1, sharex=True, sharey = True, subplot_kw={'aspect': 'equal'}, figsize = (12,4))

        else: 
            fig, axes = plt.subplots(1,len(intervals)-1, sharex=True, sharey = True, subplot_kw={'aspect': 'equal'}, figsize = (12,6))
        cmap = plt.cm.jet           
        cmap.set_bad('white')
        
        plt.subplots_adjust(hspace=0, wspace=0)
        
        t = 1
        for k in range(len(intervals)-1):
            if obs == True:
                obs_d = self.cube['obs']
                
                index_left = (np.abs(np.asarray(self.cube['v']) - intervals[k])).argmin()
                index_right = (np.abs(np.asarray(self.cube['v']) - intervals[k+t])).argmin()
                
                obs_fl_map = np.nansum(obs_d[index_left:index_right], axis=0)
                
                print('Data interval:', int(self.cube['v'][index_left]), int(self.cube['v'][index_right]))

                

            lim = (self.vlos >= intervals[k]) & (self.vlos <= intervals[k+t]) # select the model velocity range to be plotted
            print('Model interval:', intervals[k], intervals[k+t])
            
            data = np.vstack([self.yobs_psf[lim], self.xobs_psf[lim]]).T # Y, X

            
            flux_maps, edges_maps = np.histogramdd(data, bins=self.cube['nbins'][1:], range=self.cube['range'][1:], weights=self.cube['weights'][lim])#self.flux[lim]* # create mom0 map of selected interval
            
            mask_flux = flux_maps == 0
            flux_maps[mask_flux] = np.nan
            not_blank = mask_flux.shape[0]*mask_flux.shape[1]
            if (obs == True) and (only_obs == False): 
                obs_fl_map[mask_flux] = np.nan 
                il_obs = axes[0][k].imshow(obs_fl_map,interpolation='nearest',  aspect='equal', origin='lower',cmap=cmap)
                il = axes[1][k].imshow(flux_maps,interpolation='nearest',  aspect='equal', origin='lower',cmap=cmap)
                if residual == True:
                    axes[2][k].imshow(obs_fl_map-flux_maps,interpolation='nearest',  aspect='equal', origin='lower',cmap=cmap)
                axes[0][k].set_title('[%.0f,%.0f]'%(intervals[k], intervals[k+t]), fontsize = 14, weight="bold")
                if residual == True: 
                    riga = 2
                    axes[riga][0].set_ylabel('RESIDUALS \n y[Pixel]', fontsize = 14)

                else: riga = 1
                axes[riga][k].set_xlabel('X[Pixel]', fontsize = 14)
                axes[0][0].set_ylabel('DATA \n y[Pixel]', fontsize = 14)
                axes[1][0].set_ylabel('MODEL \n y[Pixel]', fontsize = 14)
                if residual == True: axes[2][0].set_ylabel('RESIDUAL \n y[Pixel]', fontsize = 14)

            elif (obs == True) and (only_obs == True):
                il = axes[k].imshow((obs_fl_map),interpolation='nearest',  aspect='equal', origin='lower',cmap=cmap,vmin=flrange[0], vmax=flrange[1])
                axes[k].set_title('[%.0f,%.0f]'%(intervals[k], intervals[k+t]), fontsize = 14, weight="bold")
                axes[k].set_xlabel('X[Pixels]', fontsize = 14)
                axes[0].set_ylabel('DATA \n y[Pixel]', fontsize = 14)

                


            elif obs == False:
                il = axes[k].imshow(flux_maps,interpolation='nearest',  aspect='equal', origin='lower',cmap=cmap, norm=colors.LogNorm())# vmin=flrange[0], vmax=flrange[1],
                axes[k].set_title('[%.0f,%.0f]'%(intervals[k], intervals[k+t]), fontsize = 14, weight="bold")
            if k == len(intervals)-2:
                
                fig.colorbar(il, ax=axes.ravel().tolist(),  label = '[erg s$^{-1}$ cm$^{-2}$]')

        t+=1
 


        

#%%
class model(utils):
    
    def __init__(self,  npt=1000000,             # number of points in simulation
                        use_seeds=True,          # use defined seeds
                        seeds = {'theta':1, 
                                 'zeta':2, 
                                 'phi':3, 
                                 'radius':4, 
                                 'vsigx':5, 
                                 'vsigy':6, 
                                 'vsigz':7, 
                                 'xpsf':8, 
                                 'ypsf':9, 
                                 'zlsf':10},
                        #
                        geometry='spherical',    # spherical or cylindrical
                        #
                        logradius=False,         # radius with logarithmic sampling?
                        radius_range=[0.0001,1], # range of radii [arcsec]
                       
                        theta_range=[[0,180]],   # range of theta angle [deg]
                        phi_range=[[0, 360.]],     # range of phi angle [deg]
                        zeta_range=[-1.,+1],       # range of z coord [arcsec]

                        flux_func=None,          # flux function func(rad, th, phi, params)
                        vel1_func=None,          # radial velocity function func(rad, th, phi, params)
                        vel2_func=None,          # theta velocity function func(rad, th, phi, params)
                        vel3_func=None,          # phi velocity function func(rad, th, phi, params)
                        vel_sigma=None,          # random velocity dispersion (either [sigx, sigy, sigz], or sig)

                        psf_sigma=None,          # beamsize [bmaj,bmin,P.A.] [arcsec]
                        lsf_sigma=None,          # sigma of lsf [km/s]

                        cube_range = [ None, None, None ], # velocity, y, x ranges
                        cube_nbins = [ None, None, None ], # velocity, y, x num bins
                        verbose = False
                  ):

        self.verbose = verbose
        
        if (geometry == 'spherical') or (geometry == 'cylindrical'):
            if self.verbose: print('You have chosen '+geometry+' geometry')  ##COSIMO
            self.geometry = geometry
        else:
            sys.exit('ERROR: choose between "spherical" or "cylindrical" geometry')
                        
            
        # Define the class params
        self.npt = npt
        self.use_seeds = use_seeds
        self.seeds = seeds
        self.radius_range = radius_range

            
        self.phi_range = phi_range
        
        # zeta is not used in spherical coords
        # theta is not used in cylindrical coords
        if self.geometry=='spherical':
            self.theta_range = theta_range
        if self.geometry=='cylindrical':
            self.zeta_range = zeta_range
            
        # assign the desider vel functions to the class
        self.logradius = logradius
        self.flux_func = flux_func
        self.vel1_func = vel1_func
        self.vel2_func = vel2_func
        self.vel3_func = vel3_func

        if vel_sigma is None:
            self.vel_sigma = None
        else:
            if isinstance(vel_sigma, int) or isinstance(vel_sigma, float):
                self.vel_sigma = np.ones(3)*vel_sigma
            elif isinstance(vel_sigma, list) and len(vel_sigma) == 3:
                self.vel_sigma = vel_sigma
            else:
                sys.exit('vel_sigma either 1 or 3 elements')


        if psf_sigma is None:
            print('No spatial convolution')
            self.psf_sigma = None
        else:
            if isinstance(psf_sigma, list) and len(psf_sigma) == 1:
                self.psf_sigma = [psf_sigma[0], psf_sigma[0], 0]
            elif isinstance(psf_sigma, list) and len(psf_sigma) == 3:
                self.psf_sigma = [ psf_sigma[0]/(2*np.sqrt(2*np.log(2))), 
                                    psf_sigma[1]/(2*np.sqrt(2*np.log(2))),
                                    np.radians(psf_sigma[2]+90) ]
            else:
                logger.critical('The PSF must be either 1 or 3 elements')
                sys.exit()
       

        if lsf_sigma is None:
            print('No spectral convolution')
            self.lsf_sigma = None
        else:
            self.lsf_sigma = lsf_sigma
        
        self.cube = { }
        self.cube['range'] = cube_range
        self.cube['nbins'] = cube_nbins
        
        self.maps = { }
        # self.cube_mod = { }

        # 
        # define clouds r, th, phi 
        #
        if self.geometry=='spherical':
            # non-uniform distribution but assign weights for histogramdd
            if self.use_seeds: np.random.seed(self.seeds['theta'])
            
            # this is used in case of a bicone
            if len(self.theta_range)>1:
                n_cones = len(self.theta_range)
                
                self.theta = np.array([0])
                for theta_range in self.theta_range:
                    theta = np.random.uniform(low=rad(theta_range[0]), high=rad(theta_range[1]), size=int(self.npt/n_cones))
                    self.theta = np.concatenate((self.theta, theta))
                self.theta = self.theta[1:]
            else: self.theta = np.random.uniform(low=rad(self.theta_range[0][0]), high=rad(self.theta_range[0][1]), size=self.npt) # generate random uniform array with npts element
                
             

        if self.geometry=='cylindrical':
            if self.use_seeds: np.random.seed(self.seeds['zeta'])
            self.zeta = np.random.uniform(low=self.zeta_range[0], high=self.zeta_range[1], size=self.npt)
     
        
        if self.use_seeds: np.random.seed(self.seeds['phi'])
        if len(self.phi_range)>1:
            n_ = len(self.phi_range)
            
            self.phi = np.array([0])
            for phi_range in self.phi_range:
                phi = np.random.uniform(low=rad(phi_range[0]), high=rad(phi_range[1]), size=int(self.npt/n_))
                self.phi = np.concatenate((self.phi,phi))
            self.phi = self.phi[1:]
        else: self.phi = np.random.uniform(low=rad(self.phi_range[0][0]), high=rad(self.phi_range[0][1]), size=self.npt) # generate random uniform array with npts element

        
        if self.geometry=='spherical':
            if self.logradius:
                if self.use_seeds: np.random.seed(self.seeds['radius'])
                radius = 10**(np.random.uniform(low=np.log10(self.radius_range[0]), high=np.log10(self.radius_range[1]), size=self.npt))
                # multiply by rad**2 to make it constant if n(logr) = const.
                self.flux_radius = radius**3*np.abs(np.sin(self.theta))/self.npt ### COSIMO: CONTROLLARE - normalizzare su interavlli raggi, theta, phi
            else:
                # non-uniform distribution but assign weights for histogramdd
                # multiply by rad to make it constant if n(r) = const.
                if self.use_seeds: np.random.seed(self.seeds['radius'])
                radius = np.random.uniform(low=self.radius_range[0], high=self.radius_range[1], size=self.npt)
                  


                self.flux_radius = radius**2*np.abs(np.sin(self.theta))/self.npt # normalizzare su interavlli raggi, theta, phi
                

        if self.geometry=='cylindrical':
            if self.logradius:
                sys.exit('logradius option in cylindrical coordinates is currently work in progress, please choose "logradius=False"...')
            else:
                # non-uniform distribution but assign weights for histogramdd               
                if self.use_seeds: np.random.seed(self.seeds['radius'])
                radius = np.random.uniform(low=self.radius_range[0],high=self.radius_range[1],size=self.npt)                
                self.flux_radius = 2.*radius/(self.radius_range[1]**2)          

        self.radius = radius
        
                
        if self.geometry=='spherical':

                
            sinth = np.sin(self.theta)
            costh = np.cos(self.theta)
            sinph = np.sin(self.phi)
            cosph = np.cos(self.phi)
            
            # u1 = u_r 
            self.u1_x = sinth*cosph
            self.u1_y = sinth*sinph
            self.u1_z = costh
            
            # u2 = u_theta 
            self.u2_x = costh*cosph
            self.u2_y = costh*sinph
            self.u2_z = -sinth
            
            # u3 = u_phi
            self.u3_x = -sinph 
            self.u3_y = cosph 
            self.u3_z = np.zeros(sinth.shape)
            
            # R = r*u_r
            self.x = self.radius*self.u1_x
            self.y = self.radius*self.u1_y
            self.z = self.radius*self.u1_z

             
        elif self.geometry=='cylindrical':
            sinph = np.sin(self.phi)
            cosph = np.cos(self.phi)
            # u1 = u_r 
            self.u1_x = cosph
            self.u1_y = sinph
            self.u1_z = np.zeros(sinph.shape)
            #  u2 = u_z
            self.u2_x = np.zeros(sinph.shape)
            self.u2_y = np.zeros(sinph.shape)
            self.u2_z = np.ones(sinph.shape)
            # u3 = u_phi 
            self.u3_x = -sinph
            self.u3_y = cosph
            self.u3_z = np.zeros(sinph.shape)
            
            # R = r*u_r+z*u_z
            self.x = self.radius*self.u1_x
            self.y = self.radius*self.u1_y
            self.z = np.copy(self.zeta)
            self.theta = np.pi/2-np.arctan(self.zeta/self.radius)


        if self.vel_sigma is not None:  
            if self.vel_sigma[0] > 0:
                if self.use_seeds: np.random.seed(self.seeds['vsigx'])
                self.vsigx = np.random.normal(loc=0.0, scale=self.vel_sigma[0], size=self.npt)
            else:
                self.vsigx = np.zeros(self.npt)
                
            if self.vel_sigma[1] > 0:
                if self.use_seeds: np.random.seed(self.seeds['vsigy'])
                self.vsigy = np.random.normal(loc=0.0, scale=self.vel_sigma[1], size=self.npt)
            else:
                self.vsigy = np.zeros(self.npt)

            if self.vel_sigma[2] > 0:
                if self.use_seeds: np.random.seed(self.seeds['vsigz'])
                self.vsigz = np.random.normal(loc=0.0, scale=self.vel_sigma[2], size=self.npt)
            else:
                self.vsigz = np.zeros(self.npt)


        if self.psf_sigma is not None:                       
            if self.use_seeds: np.random.seed(self.seeds['xpsf'])
            xpsf = np.random.normal(loc=0.0, scale=self.psf_sigma[0], size=self.npt)
            
            
            if self.use_seeds: np.random.seed(self.seeds['ypsf'])
            ypsf = np.random.normal(loc=0.0, scale=self.psf_sigma[1], size=self.npt)
            

            if self.psf_sigma[2] != 0:
                # beam rotation
                cospa = np.cos(self.psf_sigma[2])
                sinpa = np.sin(self.psf_sigma[2])
                self.xpsf = xpsf*cospa-ypsf*sinpa
                self.ypsf = xpsf*sinpa+ypsf*cospa
            else:
                self.xpsf = xpsf
                self.ypsf = ypsf
        else:
            self.xpsf = 0
            self.ypsf = 0

        
        if self.lsf_sigma is not None:
            if self.use_seeds: np.random.seed(self.seeds['zlsf'])
            self.zlsf = np.random.normal(loc=0.0, scale=self.lsf_sigma, size=self.npt)
            
        else:
            self.zlsf = 0

    # FLUX distribution
    def set_flux(self, funcpars):

        if self.flux_func is not None:
            if self.geometry == 'spherical':
                self.flux = self.flux*self.flux_func(self.radius, self.theta, self.phi, funcpars)
            else:
                self.flux = self.flux*self.flux_func(self.radius, self.zeta,  self.phi,  funcpars)

    # VELOCITIES
    # v1 = vr 
    def set_vel1(self, funcpars):

        if self.vel1_func is not None:
            if self.geometry == 'spherical':
                self.vel1 = self.vel1_func(self.radius, self.theta, self.phi, funcpars)
            else:
                self.vel1 = self.vel1_func(self.radius, self.zeta,  self.phi, funcpars)

            self.velx += self.vel1*self.u1_x
            self.vely += self.vel1*self.u1_y
            self.velz += self.vel1*self.u1_z
            
    # v1 = vtheta, vzeta      
    def set_vel2(self, funcpars):

        if self.vel2_func is not None:
            if self.geometry == 'spherical':
                self.vel2 = self.vel2_func(self.radius, self.theta, self.phi, funcpars)
            else:
                self.vel2 = self.vel2_func(self.radius, self.zeta,  self.phi, funcpars)
                
            self.velx += self.vel2*self.u2_x
            self.vely += self.vel2*self.u2_y
            self.velz += self.vel2*self.u2_z

    # v3 = vphi (spherical)
    def set_vel3(self, funcpars):

        if self.vel3_func is not None:
            if self.geometry == 'spherical':
                self.vel3 = self.vel3_func(self.radius, self.theta, self.phi, funcpars)
            else:
                self.vel3 = self.vel3_func(self.radius, self.zeta,  self.phi,  funcpars)
                
            self.velx += self.vel3*self.u3_x
            self.vely += self.vel3*self.u3_y
            self.velz += self.vel3*self.u3_z

    def set_vrand(self):  
        
        if self.vel_sigma is not None:
            self.velx += self.vsigx
            self.vely += self.vsigy
            self.velz += self.vsigz


    def generate_clouds(self, flux_pars=None, flux_pars_disk = None, vel1_pars=None, vel2_pars=None, vel3_pars=None, vel_disk_pars=None):
        """
        This function generates the clouds
        i.e., assign them a standard flux and velocity in 3D space
             then build the geometry in the source ref frame
        """
        


        # cloud fluxes
        self.flux = np.copy(self.flux_radius) 

        self.set_flux(flux_pars) # set fluxes 

        # cloud velocities
        self.velx = 0        
        self.vely = 0        
        self.velz = 0  

        self.set_vel1(vel1_pars)
        
        self.set_vel2(vel2_pars)

        self.set_vel3(vel3_pars)
        
        self.set_vrand()
 
        self.Pref = np.stack([self.x, self.y, self.z])
        self.Vref = np.stack([self.velx, self.vely, self.velz])

    
    def add_model(self, mod, weight=None, spshift=None):
        """"
        This function allows to combine a pre-existing model with many more
        """
        
        if weight is not None:
            norm = weight/mod.flux.sum()*self.flux.sum()   
        else:
            norm = 1.0
            
        self.radius = np.concatenate([self.radius, mod.radius])

        self.theta = np.concatenate([self.theta, mod.theta])
        
        if self.geometry=='spherical':
            pass
        elif self.geometry=='cylindrical':
            self.zeta = np.concatenate([self.zeta, mod.zeta])
        else:
            sys.exit('Problem with model geometry')
        
        if spshift is not None:# Ideally this would be a spatial shift among models
            sys.exit('Spatial shift not implemented yet')  
            
        self.phi = np.concatenate([self.phi,mod.phi])

        self.flux = np.concatenate([self.flux,norm*mod.flux])

        self.x = np.concatenate([self.x,mod.x])
        self.y = np.concatenate([self.y,mod.y])
        self.z = np.concatenate([self.z,mod.z])

        self.xpsf = np.concatenate([self.xpsf,mod.xpsf])
        self.ypsf = np.concatenate([self.ypsf,mod.ypsf])
        self.zlsf = np.concatenate([self.zlsf,mod.zlsf])

        #initialize velocities
        self.velx = np.concatenate([self.velx,mod.velx])        
        self.vely = np.concatenate([self.vely,mod.vely])        
        self.velz = np.concatenate([self.velz,mod.velz])        

        self.vsigx = np.concatenate([self.vsigx,mod.vsigx])  
        self.vsigy = np.concatenate([self.vsigy,mod.vsigy])
        self.vsigz = np.concatenate([self.vsigz,mod.vsigz])
        
        
        xref = np.concatenate([self.Pref[0,:],mod.Pref[0,:]])
        yref = np.concatenate([self.Pref[1,:],mod.Pref[1,:]])
        zref = np.concatenate([self.Pref[2,:],mod.Pref[2,:]])
        
        vxref = np.concatenate([self.Vref[0,:],mod.Vref[0,:]])
        vyref = np.concatenate([self.Vref[1,:],mod.Vref[1,:]])
        vzref = np.concatenate([self.Vref[2,:],mod.Vref[2,:]])
        
        
        self.Pref = np.stack([xref, yref, zref])
        self.Vref = np.stack([vxref, vyref, vzref])

        
        
        
        self.xobs = np.concatenate((self.xobs, mod.xobs))
        self.yobs = np.concatenate((self.yobs, mod.yobs))
        self.zobs = np.concatenate((self.zobs, mod.zobs))
        
        self.xobs_psf = np.concatenate((self.xobs_psf, mod.xobs_psf))
        self.yobs_psf = np.concatenate((self.yobs_psf, mod.yobs_psf))
        self.vlos_lsf = np.concatenate((self.vlos_lsf, mod.vlos_lsf))
        self.vlos = np.concatenate((self.vlos, mod.vlos))
 
        self.npt = self.radius.size
        
    def observe_clouds(self, xycenter=None, alpha=None, beta=None, gamma=None, vsys=None, xycenter_disk=None, alpha_disk=None, beta_disk=None, gamma_disk=None):
        """
        Here we 'observe' the model.
        Apply the Euler rotation to transform the model from the source 
        ref frame to the sky, i.e. as the observer sees it
        """


        if xycenter is None:
            sys.exit('Define model origin on sky')
        else:
            self.xycenter = xycenter
            
        if alpha is None or beta is None or gamma is None:
            sys.exit('Define Euler angles of model')
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        
        if vsys is None:
            sys.exit('Define systemic velocity along line of sight')
        self.vsys = vsys

        Pobs = np.dot(rot.eulermat(rad(alpha), rad(beta), rad(gamma)), self.Pref)

        self.xobs = Pobs[0,:]+xycenter[0]
        self.yobs = Pobs[1,:]+xycenter[1]
        self.zobs = Pobs[2,:]

        self.xobs_psf = self.xobs+self.xpsf
        self.yobs_psf = self.yobs+self.ypsf
            
        Zobs = np.array([0., 0., -1.]).transpose() # vector in observed frame (line of sight)

        Zref = np.dot(rot.eulermat_inverse(rad(alpha), rad(beta), rad(gamma)), Zobs)
        
        # Vlos, added systemic velocity
        self.vlos = np.dot(Zref,self.Vref)+vsys

        # spectral convolution        
        self.vlos_lsf = self.vlos+self.zlsf

        

    def generate_cube(self, weights=None):
        """
        Here we create the cube with clouds, i.e. a simple histogram in 3D
        if weights == None the clouds have std fluxes
        if weights != Nonethe model clouds are weighted with the observed flux
        """

        
        data = np.vstack([self.vlos_lsf, self.yobs_psf, self.xobs_psf]).T # V, Y, X
        safe_range = []
        for r in self.cube["range"]:
            r0, r1 = float(r[0]), float(r[1])
            if not np.isfinite(r0) or not np.isfinite(r1):
                raise ValueError(f"Non-finite histogram range encountered: {r}")
            if r1 < r0:
                r0, r1 = r1, r0
            if r1 == r0:
                raise ValueError(f"Degenerate histogram range encountered: {r}")
            safe_range.append([r0, r1])


        if weights is None:

            hist, edges = np.histogramdd(data, bins=self.cube['nbins'], range=safe_range)

            self.cube['hist'] = hist
            self.cube['data'] = hist
            
            self.cube['vedges'] = edges[0]
            self.cube['yedges'] = edges[1]
            self.cube['xedges'] = edges[2]

            self.cube['vindex'] = np.searchsorted(self.cube['vedges'], self.vlos_lsf, side='left')-1 #Find the indices into a sorted array self.cube['vedges'] such that, if the corresponding elements in self.vlos_lsf were inserted before the indices, the order of a would be preserved
            self.cube['yindex'] = np.searchsorted(self.cube['yedges'], self.yobs_psf, side='left')-1
            self.cube['xindex'] = np.searchsorted(self.cube['xedges'], self.xobs_psf, side='left')-1

            self.cube['v'] = (edges[0][1:] + edges[0][:-1]) / 2.
            self.cube['y'] = (edges[1][1:] + edges[1][:-1]) / 2.
            self.cube['x'] = (edges[2][1:] + edges[2][:-1]) / 2.
    
            self.cube['dv'] = (self.cube['v'][1:]-self.cube['v'][:-1])[0]
            self.cube['dy'] = (self.cube['y'][1:]-self.cube['y'][:-1])[0]
            self.cube['dx'] = (self.cube['x'][1:]-self.cube['x'][:-1])[0]
    
            self.cube['nv'], self.cube['ny'], self.cube['nx'] = self.cube['hist'].shape
            
            self.cube['vextent'] = [edges[0][0], edges[0][-1]]
            self.cube['yextent'] = [edges[1][0], edges[1][-1]]
            self.cube['xextent'] = [edges[2][0], edges[2][-1]]

        else:      
            # If weight is not None, the clouds are weighted with flux
            cube, edges = np.histogramdd(data,  bins=self.cube['nbins'], range=safe_range, weights=weights)
            self.cube['data'] = cube 
            self.cube['nv'], self.cube['ny'], self.cube['nx'] = self.cube['data'].shape
                
 
    def weight_cube(self, obscube):
        """
        
        Crucial part of weighting the model clouds.
        
        Look at the data flux in each spectral and spatial bin
        and distribuite it among the model clouds in the corresponding bin
        """
        

            
        weights = np.zeros(self.npt)

        vindex = self.cube['vindex']
        yindex = self.cube['yindex']
        xindex = self.cube['xindex']
        nv, ny, nx = self.cube['hist'].shape
        
        w = (vindex > -1) & (vindex < nv) & (yindex > -1) & (yindex < ny) & (xindex > -1) & (xindex < nx)
        weights[w] = obscube[vindex[w], yindex[w], xindex[w]]/self.cube['data'][vindex[w], yindex[w], xindex[w]]
        self.cube['obs'] = obscube
        self.cube['weights'] = weights
        


#%%
class observed(utils):
    
    def __init__(self, data,  error=None, crval=None, cdelt=None, crpix=None, wlref=None,
                  velmap = None, sigmap=None, fluxmap=None, frequency = None, vel_array = None, new_method_mom_maps = None):
 
        self.cube = {}
        
        self.crval = crval
        self.cdelt = cdelt
        self.crpix = crpix
        
        self.cube["crval"] = crval
        self.cube["cdelt"] = cdelt
        self.cube["crpix"] = crpix

        self.cube['data'] = data
        
        # the uncertainty on the data flux is still to be implemented
        if error is not None:
            self.cube['err'] = error
        else:
            self.cube['err'] = self.cube['data']*0.+1

        # copy the data shape for the cube
        nv, ny, nx = self.cube['nv'], self.cube['ny'], self.cube['nx'] = self.cube['data'].shape 

        self.cube['nbins'] = self.cube['data'].shape

        #  inizialize vector for the axis
        px = np.arange(0, nx, 1)+1         
        py = np.arange(0, ny, 1)+1
        pv = np.arange(0, nv, 1)+1
        
        # reconstruct the axis 
        x = crval[2]+cdelt[2]*(px-crpix[2])        
        y = crval[1]+cdelt[1]*(py-crpix[1])
        v = crval[0]+cdelt[0]*(pv-crpix[0])



        if frequency is not None: 
            # v is actually the frequency vector if frequency == true
            v = (3e5 / v)* 1e13#in AA
        
        if wlref is not None:
            v = (v/wlref-1.0)*3e5
       
        self.cube['x'] = x
        self.cube['y'] = y
        self.cube['v'] = v

        self.cube['dx'] = cdelt[2]
        self.cube['dy'] = cdelt[1]
        self.cube['dv'] = cdelt[0]
        
        # find the range of the axis
        xrange = self.findrange(x)      
        yrange = self.findrange(y)
        vrange = self.findrange(v)

        
        self.cube['range'] = [ vrange, yrange, xrange ]

        # set the extension of each dimension 
        # with the central coordinates of the pixels of the edges
        self.cube['xextent'] = [ xrange[0]-cdelt[2]/2., xrange[1]+cdelt[2]/2. ]  
        self.cube['yextent'] = [ yrange[0]-cdelt[1]/2., yrange[1]+cdelt[1]/2. ] 
        self.cube['vextent'] = [ vrange[0]-cdelt[0]/2., vrange[1]+cdelt[0]/2. ]  
       

        self.maps = { }
        # if fluxmap is None or velmap is None or sigmap is None: 
        # call fuction that makes the mom-0,1,2-maps
        #     self.kin_maps_cube(fluxthr = 1e-20)                                    
        
        
        if new_method_mom_maps is None: 
            new_method_mom_maps = False

        if fluxmap is None and velmap is None and sigmap is  None: 
            self.kin_maps_cube(fluxthr = 1e-30, vel_array =vel_array, new_method_mom_maps = new_method_mom_maps) 


        if fluxmap is None and velmap is not None and sigmap is not None:  
            self.maps['vel'] = fits.getdata(velmap)
            self.maps['sig'] = fits.getdata(sigmap)
            self.kin_maps_cube(fluxthr = 1e-30, velocity = True, vel_disp=True) 
        if velmap is None and fluxmap is not None and sigmap is not None :
            self.maps['flux'] = fits.getdata(fluxmap)              
            self.maps['sig'] = fits.getdata(sigmap)
            self.kin_maps_cube(fluxthr = 1e-30, flux = True, vel_disp=True) 
        if sigmap is None and fluxmap is not None and velmap is not None:
            self.maps['flux'] = fits.getdata(fluxmap)              
            self.maps['vel'] = fits.getdata(velmap)
            self.kin_maps_cube(fluxthr = 1e-30, flux = True, velocity=True) 

            
        if sigmap is not None and fluxmap is not None and velmap is not None:
            # In case you want to use pre-computed moment maps
            # instead of making the model create them
            self.maps['flux'] = fits.getdata(fluxmap)             
            self.maps['vel'] = fits.getdata(velmap)
            self.maps['sig'] = fits.getdata(sigmap)




    

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
#%% functions for the main code, read data, WCS, find center etc..


def load_cube_and_wcs(datadir, name_data, prefer_hdu1=True):
    """
    Read FITS cube robustly:
      - If multiple HDUs and prefer_hdu1=True -> use HDU=1 else HDU=0
      - Sanitize header (WCS units, BLANK, RADECSYS->RADESYSa)
      - Build 2D WCS (celestial) + full WCS
    Returns:
      obscube, obshead, wcs2d, wcs_large, hdu_index
    """
    path = os.path.join(datadir, name_data)

    with fits.open(path) as hdul:
        hdu_index = 1 if (prefer_hdu1 and len(hdul) > 1) else 0
        data = hdul[hdu_index].data
        head = hdul[hdu_index].header.copy()

    if data is None:
        raise ValueError(f"Selected HDU {hdu_index} has no data. Available HDUs: {len(hdul)}")

    # sanitize header once
    head = _sanitize_fits_header(head)

    # WCS objects
    wcs2d = WCS(head, naxis=2)
    wcs_large = WCS(head)

    return data, head, wcs2d, wcs_large, hdu_index


def pixel_scale_arcsec(wcs_large):
    """Mean pixel scale in arcsec/pix from celestial WCS."""
    pixscale_deg = proj_plane_pixel_scales(wcs_large.celestial) * u.deg
    return pixscale_deg.to(u.arcsec).mean().value


def spectral_axis_from_header(obshead, n_spec):
    """
    Build spectral wavelength array from FITS header (axis 3):
      CRVAL3, CRPIX3, CDELT3 or CD3_3, and CUNIT3.
    Returns:
      wl_m (Quantity, meters), spec_unit (original unit), spec_coord (Quantity in spec_unit)
    """
    crval3 = float(obshead.get("CRVAL3"))
    crpix3 = float(obshead.get("CRPIX3"))
    cdelt3 = obshead.get("CD3_3", obshead.get("CDELT3"))
    if cdelt3 is None:
        raise KeyError("Could not find CD3_3 or CDELT3 for spectral axis in header.")
    cdelt3 = float(cdelt3)

    cunit3_raw = str(obshead.get("CUNIT3", "")).strip()
    cunit3_map = {
        "MICRON": "um",
        "micron": "um",
        "UM": "um",
        "DEGREE": "deg",
        "degree": "deg",
    }
    cunit3 = cunit3_map.get(cunit3_raw, cunit3_raw)

    try:
        spec_unit = u.Unit(cunit3) if cunit3 else u.one
    except Exception:
        raise ValueError(f"Unrecognized spectral unit CUNIT3='{cunit3_raw}' after sanitizing -> '{cunit3}'")

    pix = np.arange(n_spec, dtype=float) + 1.0  # FITS is 1-based
    spec_coord = (crval3 + (pix - crpix3) * cdelt3) * spec_unit

    try:
        wl_m = spec_coord.to(u.m)
    except Exception:
        raise ValueError(
            f"Spectral axis unit '{spec_unit}' is not convertible to meters. "
            "Header likely does not store wavelength on axis 3."
        )
    return wl_m, spec_unit, spec_coord


def find_systemic_channel(obscube, wl_m, wavelength_line, wavelength_line_unit, redshift, vwin_kms=3000.0):
    """
    Determine systemic wavelength from cube:
      - sum spectrum
      - search peak within +/- vwin_kms around expected observed wavelength
      - fallback to global peak
    Returns:
      i0 (int), wl_sys_m (Quantity), wl0_expected_m (Quantity)
    """
    spec = np.nansum(obscube, axis=(1, 2))
    spec = np.where(np.isfinite(spec), spec, 0.0)

    wl_rest_m = _parse_wavelength_input(wavelength_line, wavelength_line_unit, u.m)
    wl0_expected_m = wl_rest_m * (1.0 + redshift)

    vwin = vwin_kms * u.km / u.s
    mask = np.abs((c * (wl_m / wl0_expected_m - 1.0)).to(u.km / u.s)) <= vwin

    if np.any(mask):
        i_local = int(np.argmax(spec[mask]))
        i0 = int(np.where(mask)[0][i_local])
    else:
        i0 = int(np.argmax(spec))

    return i0, wl_m[i0], wl0_expected_m


def velocity_axis_from_wavelength(wl_m, wl_sys_m):
    """Relative velocity array around wl_sys."""
    vel = (c * (wl_m / wl_sys_m - 1.0)).to(u.km / u.s)
    return vel.to_value(u.km / u.s)


def pick_center(wcs2d, obscube, vel_kms, agn_ra, agn_dec, center_mode, center_xy_manual,
                flux_kwargs=None, kin_kwargs=None):
    """
    Decide center:
      1) world coords if both provided
      2) manual pixels if center_mode is None
      3) flux / kinematic otherwise

    Returns:
      x_cen, y_cen, sx, sy, center_used
    """
    flux_kwargs = flux_kwargs or dict(vwin_kms=300.0, box=7)
    kin_kwargs  = kin_kwargs  or dict(vwin_kms=500.0, box=7, flux_q=70)

    use_world_center = (agn_ra is not None) and (agn_dec is not None)

    if center_xy_manual is not None:
        if use_world_center or (center_mode is not None):
            raise ValueError("center_xy_manual is set -> set agn_ra=None, agn_dec=None and center_mode=None.")

    if use_world_center:
        if isinstance(agn_ra, str):
            agn_coords = SkyCoord(agn_ra, agn_dec, unit=(u.hourangle, u.deg))
        else:
            agn_coords = SkyCoord(ra=float(agn_ra) * u.deg, dec=float(agn_dec) * u.deg)

        w = wcs2d.celestial if hasattr(wcs2d, "celestial") else wcs2d

        # ---------------------------------------------------------
        # Check whether the WCS is usable for RA/Dec -> pixel conversion
        # ---------------------------------------------------------
        w_naxis = getattr(w, "naxis", None)
        w_has_celestial = getattr(w, "has_celestial", False)

        if (w is None) or (w_naxis is None) or (w_naxis < 2) or (not w_has_celestial):
            raise RuntimeError(
                "Your target has an empty or invalid celestial WCS, so the center cannot "
                "be defined from RA/Dec coordinates. Set agn_ra: null and dec_ra: null, then "
                "use another method to define the center: either provide pixel coordinates "
                "with center_xy_manual (e.g. [x_cen, y_cen] or choose a different center_mode (either flux or kinematic)."
            )

        try:
           x_pix, y_pix = w.world_to_pixel(agn_coords)
        except Exception:
            try:
                x_pix, y_pix = w.world_to_pixel_values(
                    agn_coords.ra.deg,
                    agn_coords.dec.deg
                )
            except Exception as e2:
                raise RuntimeError(
                    "Your target has an unusable celestial WCS, so the center cannot "
                    "be defined from RA/Dec coordinates. "
                    "Use another method to define the center: either provide pixel coordinates "
                    "with center_xy_manual or choose a different center_mode."
                ) from e2

       

        return float(x_pix), float(y_pix), np.nan, np.nan, "world"

    if center_mode is None:
        if center_xy_manual is None:
            raise ValueError("agn_ra/agn_dec are None and center_mode is None -> set center_xy_manual=(x_pix,y_pix).")
        return float(center_xy_manual[0]), float(center_xy_manual[1]), 0.0, 0.0, "manual"

    cm = str(center_mode).lower()
    cube3, _ = standardize_cube_to_spec_yx(obscube, n_spec=len(vel_kms))
    if cm == "flux":
        x, y, sx, sy = find_center_from_flux_peak_with_unc(cube3, vel_kms, **flux_kwargs)
        return x, y, sx, sy, "flux"

    if cm == "kinematic":
        x, y, sx, sy = find_center_from_kinematic_with_unc(cube3, vel_kms, **kin_kwargs)
        return x, y, sx, sy, "kinematic"

    raise ValueError("center_mode must be 'flux', 'kinematic', or None.")


def observed_wcs_params_from_vel(vel_kms, i0, x_cen, y_cen, pixscale, nrebin):
    """
    Build crpix/crval/cdelt for observed.
    Preserve the actual velocity axis defined in vel_kms.
    Returns: crpix, crval, cdelt, dv, ref_pix_0based, x0_bin, y0_bin
    """
    dv = float(np.nanmedian(np.diff(vel_kms)))
    if not np.isfinite(dv) or dv == 0:
        raise ValueError("Could not determine dv from velocity vector.")
    dv = np.sign(vel_kms[-1] - vel_kms[0]) * abs(dv)

    ref_pix_0based = int(i0)
    crpix_spec = float(ref_pix_0based + 1)  # FITS is 1-based
    crval_spec = float(vel_kms[ref_pix_0based])

    x0_bin = float(x_cen / nrebin)
    y0_bin = float(y_cen / nrebin)

    crpix = [crpix_spec, float(y0_bin + 1), float(x0_bin + 1)]
    crval = [crval_spec, 0.0, 0.0]
    cdelt = [dv, pixscale * nrebin, pixscale * nrebin]

    return crpix, crval, cdelt, dv, ref_pix_0based, x0_bin, y0_bin

def _sanitize_fits_header(header):
    """
    Remove/normalize known problematic FITS keywords that trigger astropy warnings
    but are irrelevant for our usage.
    """
    hdr = header.copy()

    # 1) Fix invalid BLANK (often BLANK='nan'). BLANK is only for integer images.
    if "BLANK" in hdr:
        try:
            int(hdr["BLANK"])
        except Exception:
            del hdr["BLANK"]

    # 2) RADECSYS deprecated -> ensure RADESYSa exists
    if "RADECSYS" in hdr and "RADESYSa" not in hdr:
        hdr["RADESYSa"] = hdr["RADECSYS"]

    # 3) Normalize non-standard unit strings that wcslib rejects
    hdr = _sanitize_wcs_units_in_header(hdr)

    return hdr

def _sanitize_wcs_units_in_header(header):
    """
    Fix non-standard WCS unit strings that wcslib rejects.
    Operates in-place and also returns the header for convenience.
    """
    unit_map = {
        "DEGREE": "deg",
        "DEGREES": "deg",
        "MICRON": "um",
        "MICRONS": "um",
        "UM": "um",
        "ANGSTROM": "Angstrom",
        "A": "Angstrom",
    }

    for k in list(header.keys()):
        if not str(k).startswith("CUNIT"):
            continue
        val = header.get(k)
        if val is None:
            continue

        s_up = str(val).strip().upper()
        if s_up in unit_map:
            header[k] = unit_map[s_up]

    return header


# --- robust handling of wavelength units (Å / nm / µm / m) ---
def _parse_wavelength_input(wl_value, wl_unit_hint, target_unit):
    """
    wl_value: float or Quantity
    wl_unit_hint: None or string (e.g. 'Angstrom','nm','um','micron','m')
    target_unit: astropy Unit (the cube spectral unit)
    Returns: wl_rest as Quantity in target_unit
    """
    if isinstance(wl_value, u.Quantity):
        return wl_value.to(target_unit)

    # float given -> need a unit
    if wl_unit_hint is None:
        # default assumption for floats: Angstrom (common for optical lines like 5006.8)
        in_unit = u.AA
    else:
        s = str(wl_unit_hint).strip().lower()
        if s in ("a", "aa", "angstrom", "ang"):
            in_unit = u.AA
        elif s in ("nm", "nanometer", "nanometre"):
            in_unit = u.nm
        elif s in ("um", "micron", "microns", "µm"):
            in_unit = u.um
        elif s in ("m", "meter", "metre"):
            in_unit = u.m
        else:
            raise ValueError(
                f"wavelength_line_unit='{wl_unit_hint}' not understood. "
                "Use one of: 'Angstrom','nm','um'/'micron','m', or pass a Quantity."
            )

    return (float(wl_value) * in_unit).to(target_unit)



def observed_wcs_params(vel_kms, i0, x_cen, y_cen, pixscale, nrebin):
    """
    Build crpix/crval/cdelt for observed.
    Preserve the actual velocity axis; use i0 as the reference pixel.
    Keep dv sign consistent with velocity array.
    """
    dv = float(np.nanmedian(np.diff(vel_kms)))
    if not np.isfinite(dv) or dv == 0:
        raise ValueError("Could not determine dv from velocity vector.")
    dv = np.sign(vel_kms[-1] - vel_kms[0]) * abs(dv)

    ref_pix_0based = int(i0)
    crpix_spec = float(ref_pix_0based + 1)  # FITS is 1-based
    crval_spec = 0.0

    x0_bin = float(x_cen / nrebin)
    y0_bin = float(y_cen / nrebin)

    crpix = [crpix_spec, float(y0_bin + 1), float(x0_bin + 1)]
    crval = [crval_spec, 0.0, 0.0]
    cdelt = [dv, pixscale * nrebin, pixscale * nrebin]

    return crpix, crval, cdelt, dv, ref_pix_0based, x0_bin, y0_bin


def _weighted_centroid_and_uncertainty(img, x0, y0, box=7):
    """
    Weighted centroid (sub-pixel) + simple uncertainty estimate.
    Uncertainty: sqrt(weighted variance)/sqrt(N_eff)
    Returns: (xc, yc, sig_x, sig_y)
    """
    ny, nx = img.shape
    x1 = max(0, int(x0) - box); x2 = min(nx, int(x0) + box + 1)
    y1 = max(0, int(y0) - box); y2 = min(ny, int(y0) + box + 1)

    cut = np.array(img[y1:y2, x1:x2], dtype=float)
    cut[~np.isfinite(cut)] = 0.0
    cut[cut < 0] = 0.0

    if cut.sum() <= 0:
        # fallback: no weights
        return float(x0), float(y0), np.nan, np.nan

    yy, xx = np.mgrid[y1:y2, x1:x2]
    w = cut
    wsum = w.sum()

    xc = (xx * w).sum() / wsum
    yc = (yy * w).sum() / wsum

    # weighted variances
    varx = (w * (xx - xc) ** 2).sum() / wsum
    vary = (w * (yy - yc) ** 2).sum() / wsum

    # effective number of samples (Kish)
    neff = (wsum ** 2) / (np.sum(w ** 2) + 1e-30)
    sig_x = np.sqrt(varx) / np.sqrt(max(neff, 1.0))
    sig_y = np.sqrt(vary) / np.sqrt(max(neff, 1.0))

    return float(xc), float(yc), float(sig_x), float(sig_y)

def find_center_from_flux_peak_with_unc(cube, vel_kms, vwin_kms=300.0, box=7):
    """
    Flux-based center: integrate within |v|<vwin_kms, pick peak, refine centroid, return uncertainty.
    Returns: xc, yc, sig_x, sig_y
    """
    sel = np.isfinite(vel_kms) & (np.abs(vel_kms) <= vwin_kms)
    if np.sum(sel) < 3:
        img = np.nansum(cube, axis=0)
    else:
        img = np.nansum(cube[sel, :, :], axis=0)

    if not np.isfinite(img).any():
        raise ValueError("Flux center failed: collapsed image is all NaN/inf.")

    y0, x0 = np.unravel_index(np.nanargmax(img), img.shape)
    return _weighted_centroid_and_uncertainty(img, x0, y0, box=box)

def find_center_from_kinematic_with_unc(cube, vel_kms, vwin_kms=500.0, box=7, flux_q=70):
    """
    Kinematic center: build mom1 from |v|<vwin_kms, score near systemic with high flux and grad,
    refine centroid on score map, return uncertainty.
    Returns: xc, yc, sig_x, sig_y
    """
    sel = np.isfinite(vel_kms) & (np.abs(vel_kms) <= vwin_kms)
    if np.sum(sel) < 3:
        sel = slice(None)

    cube_sel = cube[sel, :, :]
    v_sel = vel_kms[sel]

    flux = np.nansum(cube_sel, axis=0)
    good = np.isfinite(flux) & (flux > 0)

    v3d = v_sel[:, None, None]
    mom1 = np.full_like(flux, np.nan, dtype=float)
    mom1[good] = np.nansum(v3d * cube_sel, axis=0)[good] / flux[good]

    gy, gx = np.gradient(mom1)
    gradmag = np.hypot(gx, gy)

    fthr = np.nanpercentile(flux[good], flux_q) if np.any(good) else np.nan
    bright = good & (flux >= fthr)

    eps = 1e-6
    score = (gradmag * flux) / (np.abs(mom1) + eps)
    score[~bright] = -np.inf

    if not np.isfinite(score).any():
        # fallback to flux method
        return find_center_from_flux_peak_with_unc(cube, vel_kms, vwin_kms=min(vwin_kms, 300.0), box=box)

    y0, x0 = np.unravel_index(np.nanargmax(score), score.shape)

    # For centroiding, use a positive weight map (score clipped at 0)
    score2 = np.array(score, dtype=float)
    score2[~np.isfinite(score2)] = 0.0
    score2[score2 < 0] = 0.0

    return _weighted_centroid_and_uncertainty(score2, x0, y0, box=box)

def _spectral_axis_index(w):
    
    axinfo = w.get_axis_types()  # list of dicts, one per WCS axis
    for j, info in enumerate(axinfo):
        if info.get("coordinate_type", "").lower() == "spectral":
            return j
    raise ValueError("No spectral axis found in WCS.")

def _spectral_world_coords(w_full, n_spec):
    
    spec_wcs = w_full.sub(["spectral"])
    spec_unit = u.Unit(spec_wcs.wcs.cunit[0]) if spec_wcs.wcs.cunit[0] else u.one

    pix = np.arange(n_spec, dtype=float)
    spec_world = spec_wcs.all_pix2world(pix, 0)

    # --- make sure it's 1D (Astropy can return (N,1) depending on version/WCS) ---
    spec_world = np.asarray(spec_world).reshape(-1)

    return spec_world * spec_unit, spec_unit, spec_wcs

def _set_linear_axis_keywords(header, axis_number_1based, crval, cdelt, crpix):
    
    n = axis_number_1based
    header[f"CRVAL{n}"] = float(crval)
    header[f"CRPIX{n}"] = float(crpix)

    cdkey = f"CD{n}_{n}"
    dkkey = f"CDELT{n}"
    if cdkey in header:
        header[cdkey] = float(cdelt)
    elif dkkey in header:
        header[dkkey] = float(cdelt)
    else:
        # if neither exists, writing CDELTn is the most common/portable fallback
        header[dkkey] = float(cdelt)


def apply_sn_mask_to_cube(obscube, sn_fits_path, SN_map, sn_thresh):

    # ---- do nothing if disabled ----
    if sn_thresh is None:
        logger.info("SN masking skipped (sn_thresh=None)")
        return obscube

    if sn_fits_path is None or SN_map is None:
        logger.info("SN masking skipped (no SN map provided)")
        return obscube

    sn_fits_path = Path(sn_fits_path) / SN_map

    with fits.open(sn_fits_path) as hdusn:
        sn2d = np.array(hdusn[0].data, dtype=float)

    if obscube.ndim != 3:
        raise ValueError(f"Cube must be 3D (nwave,ny,nx). Got {obscube.shape}")

    nv, ny, nx = obscube.shape

    if sn2d.shape != (ny, nx):
        raise ValueError(
            f"SN map shape {sn2d.shape} != cube spatial shape {(ny, nx)}"
        )

    mask2d = (~np.isfinite(sn2d)) | (sn2d < sn_thresh)

    obscube[:, mask2d] = np.nan

    return obscube

def _parse_cunit3_to_wavelength_unit(cunit3):
    """
    Map many common FITS CUNIT3 spellings to an astropy wavelength unit.
    Raises ValueError if it can't interpret it as a wavelength unit.
    """
    if cunit3 is None:
        raise ValueError("Header missing CUNIT3; cannot determine spectral unit.")

    s = str(cunit3).strip()
    s_low = s.lower().replace(" ", "").replace("_", "")

    # Common aliases
    aliases = {
        "a": u.AA, "å": u.AA, "aa": u.AA, "angstrom": u.AA, "angstroms": u.AA,
        "ang": u.AA, "angs": u.AA,
        "nm": u.nm, "nanometer": u.nm, "nanometers": u.nm,
        "um": u.um, "µm": u.um, "micron": u.um, "microns": u.um,
        "mm": u.mm,
        "cm": u.cm,
        "m": u.m, "meter": u.m, "meters": u.m,
    }
    if s_low in aliases:
        return aliases[s_low]

    # Try letting astropy parse it (works for many strings like 'm', 'nm', 'um', etc.)
    try:
        unit = u.Unit(s)
    except Exception as e:
        raise ValueError(f"Unrecognized CUNIT3='{cunit3}' (not parseable).") from e

    # Ensure it's a length unit
    if not unit.is_equivalent(u.m):
        raise ValueError(f"CUNIT3='{cunit3}' parsed as {unit}, which is not a wavelength/length unit.")
    return unit


def _get_linear_spectral_wcs_params(header):
    """
    Return (crval3, cdelt3, crpix3) as floats from FITS header,
    supporting CD3_3 or CDELT3, and PC3_3 if needed.

    Assumes axis is linear in the spectral coordinate.
    """
    crval3 = header.get("CRVAL3", None)
    crpix3 = header.get("CRPIX3", None)
    if crval3 is None or crpix3 is None:
        raise KeyError("Header missing CRVAL3 and/or CRPIX3 for spectral axis.")

    # Step can come from CD3_3, or CDELT3, or (PC3_3 * CDELT3)
    if "CD3_3" in header:
        cdelt3 = header["CD3_3"]
    else:
        cdelt3 = header.get("CDELT3", None)
        if cdelt3 is None:
            raise KeyError("Header missing CD3_3 and CDELT3; cannot build spectral axis.")

        # Apply PC matrix element if present (common WCS form)
        pc33 = header.get("PC3_3", 1.0)
        cdelt3 = pc33 * cdelt3

    return float(crval3), float(cdelt3), float(crpix3)


def vrot(rad, theta, phi, vrotpars): 
    Rd = vrotpars[0]
    Mdyn = vrotpars[1]
    scale = vrotpars[2]
    rkpc = rad*scale
    vel = rc.vel_disk(rkpc, Rd, Mdyn, Rext=5.0, gasSigma=0)
    return vel


def vout(rad, theta, phi, voutpars): 
    return voutpars[0]
# def vexp(rad, theta, phi, voutpars):
#     return voutpars[0]*(np.sqrt(rad))
def fexpo(rad,theta, phi,fluxpars):     
    f0 = fluxpars[0]
    r0 = fluxpars[1]
    return f0*np.exp(-rad/r0)
def sp_pl_compare(data_cube,
                  cube_pre,
                  cube_post,
                  vel,
                  mask=None,
                  normalize=True):
    
    if isinstance(vel, u.Quantity):
        vel_plot = vel.to(u.km/u.s).value
        vel_unit = r"km s$^{-1}$"
    else:
        vel_plot = vel
        vel_unit = r"km s$^{-1}$"
    if mask is not None:
        data_cube  = np.where(mask, data_cube,  np.nan)
        cube_pre   = np.where(mask, cube_pre,   np.nan)
        cube_post  = np.where(mask, cube_post,  np.nan)
    sp_data = np.nansum(data_cube, axis=(1,2))
    sp_pre  = np.nansum(cube_pre, axis=(1,2))
    sp_post = np.nansum(cube_post, axis=(1,2))

    if normalize:
        def safe_norm(x):
            m = np.nanmax(np.abs(x))
            return x/m if m > 0 else x

        sp_data = safe_norm(sp_data)
        sp_pre  = safe_norm(sp_pre)
        sp_post = safe_norm(sp_post)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=250)
    ax.plot(vel_plot, sp_data,  label='Data',      color='black', lw=3)
    ax.plot(vel_plot, sp_pre,   label='Model pre-weights',  color='orange', lw=2, ls='--')
    ax.plot(vel_plot, sp_post,  label='Model post-weights', color='red', lw=2)
    ax.set_ylabel('Flux [A.U.]', fontsize=24)
    ax.set_xlabel(f'Velocity [{vel_unit}]', fontsize=24)
    ax.tick_params(axis='both', direction='in', labelsize=22, width=2)
    ax.legend(fontsize=16)
    plt.tight_layout()
    #plt.show()
    
def pa180_circ_mean_std(pa_deg):
    """
    Circular mean/std for angles with 180° periodicity.
    Returns (mean_pa_deg in [0,180), std_deg).
    """
    a = np.deg2rad(2.0 * np.asarray(pa_deg))  # map 0-180 -> 0-360
    C = np.nanmean(np.cos(a))
    S = np.nanmean(np.sin(a))
    mean2 = np.arctan2(S, C)                  # mean angle on 0-2pi
    mean_pa = (np.rad2deg(mean2) / 2.0) % 180.0

    R = np.hypot(C, S)
    # circular std on doubled angles, then halve it
    # guard R ~ 0
    if R <= 0:
        return mean_pa, np.inf
    std2 = np.sqrt(-2.0 * np.log(R))          # in radians (for doubled angle)
    std_pa = np.rad2deg(std2) / 2.0
    return mean_pa, std_pa

def _pa_from_weighted_pca(coords_xy, vv, ww):
    """Return PA in [0,180) from weighted PCA of coords (pixel units), weighted by ww."""
    mean = np.average(coords_xy, axis=0, weights=ww)
    coords_w = coords_xy - mean
    cov = np.cov(coords_w.T, aweights=ww)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]
    pa_grad_rad = np.arctan2(vec[0], vec[1])
    PA_b = (np.degrees(pa_grad_rad) + 90.0) % 180.0
    return PA_b

def _block_bootstrap_indices(mask2d, block_size, rng, n_samples_target):
    """
    Return a 1D boolean mask selecting pixels by sampling blocks (with replacement).
    mask2d: valid pixels
    block_size: side length in pixels (int)
    n_samples_target: approximate number of valid pixels to collect
    """
    ny, nx = mask2d.shape
    bs = int(max(1, block_size))

    sel = np.zeros_like(mask2d, dtype=bool)
    n_valid = int(mask2d.sum())
    if n_valid == 0:
        return sel

    # sample blocks until we collect ~n_samples_target valid pixels
    collected = 0
    tries = 0
    max_tries = 50_000  # avoid infinite loops in pathological masks

    while collected < n_samples_target and tries < max_tries:
        tries += 1
        y0 = rng.integers(0, ny)
        x0 = rng.integers(0, nx)

        y1 = min(ny, y0 + bs)
        x1 = min(nx, x0 + bs)

        block = mask2d[y0:y1, x0:x1]
        if not block.any():
            continue

        sel[y0:y1, x0:x1] |= block
        collected = int(sel.sum())

    return sel

def velocity_axis_rezero_to_systemic(vel_kms, i0):
    """
    Shift vel_kms so that vel_kms[i0] == 0.
    """
    v0 = float(vel_kms[i0])
    return vel_kms - v0


def find_systemic_channel_from_vel(obscube, vel_kms, vwin_kms=3000.0, spec_axis=None):
    """
    Choose systemic channel i0 from an integrated spectrum, restricted to |v|<vwin_kms if possible.

    Works for cubes shaped like:
      - (n_spec, ny, nx)
      - (ny, nx, n_spec)
      - (stokes, n_spec, ny, nx)
      - etc.

    Parameters
    ----------
    obscube : ndarray
        Data cube.
    vel_kms : array-like
        Velocity axis (km/s) with length n_spec.
    vwin_kms : float
        Search window around 0 km/s for finding the line peak.
    spec_axis : int or None
        If None, infer which axis is spectral by matching len(vel_kms) to cube shape.

    Returns
    -------
    i0 : int
        Index of systemic channel along the spectral axis.
    """
    vel_kms = np.asarray(vel_kms, dtype=float)
    n_spec = vel_kms.size

    # --- infer spectral axis if not provided ---
    if spec_axis is None:
        matches = [ax for ax, s in enumerate(obscube.shape) if s == n_spec]
        if len(matches) == 0:
            raise ValueError(
                f"Could not infer spectral axis: len(vel_kms)={n_spec} does not match any cube dimension {obscube.shape}."
            )
        if len(matches) > 1:
            # ambiguous but uncommon; pick the first match
            spec_axis = matches[0]
        else:
            spec_axis = matches[0]

    # --- collapse cube to a 1D spectrum along spectral axis ---
    sum_axes = tuple(ax for ax in range(obscube.ndim) if ax != spec_axis)
    spec = np.nansum(obscube, axis=sum_axes)

    # Ensure 1D and correct length
    spec = np.asarray(spec).reshape(-1)
    if spec.size != n_spec:
        raise ValueError(
            f"Integrated spectrum has size {spec.size}, expected {n_spec}. "
            f"Cube shape={obscube.shape}, spec_axis={spec_axis}."
        )

    # Clean NaNs/Infs
    spec = np.where(np.isfinite(spec), spec, 0.0)

    # --- select window in velocity space ---
    sel = np.isfinite(vel_kms) & (np.abs(vel_kms) <= float(vwin_kms))

    # If window is too small (e.g. only 0 or 1 channel), fallback to full spectrum
    if np.count_nonzero(sel) < 3:
        return int(np.argmax(spec))

    # Find peak within window
    idx_in_window = np.where(sel)[0]
    i_rel = int(np.argmax(spec[sel]))
    i0 = int(idx_in_window[i_rel])

    return i0

def standardize_cube_to_spec_yx(cube, n_spec, keep_first_if_ambiguous=True):
    """
    Return a 3D cube shaped (n_spec, ny, nx) from an arbitrary FITS cube.

    Handles typical cases:
      - (n_spec, ny, nx)
      - (ny, nx, n_spec)
      - (stokes, n_spec, ny, nx) with stokes=1
      - (n_spec, stokes, ny, nx) with stokes=1
      - etc.

    Parameters
    ----------
    cube : ndarray
    n_spec : int
        Expected length of spectral axis.
    keep_first_if_ambiguous : bool
        If multiple axes match n_spec, pick the first.

    Returns
    -------
    cube3 : ndarray
        3D cube (n_spec, ny, nx)
    spec_axis : int
        Which axis in the original (squeezed) cube was spectral
    """
    c = np.asarray(cube)

    # Drop trivial singleton axes (ALMA often has Stokes=1)
    c = np.squeeze(c)

    if c.ndim < 3:
        raise ValueError(f"Cube has ndim={c.ndim} after squeeze, expected >=3. Shape={c.shape}")

    # Find spectral axis by matching length
    matches = [ax for ax, s in enumerate(c.shape) if s == int(n_spec)]
    if len(matches) == 0:
        raise ValueError(f"Cannot find spectral axis of length {n_spec} in cube shape {c.shape}")
    if len(matches) > 1:
        spec_axis = matches[0] if keep_first_if_ambiguous else matches
    else:
        spec_axis = matches[0]

    # Move spectral axis to the front: (n_spec, ...)
    c = np.moveaxis(c, spec_axis, 0)

    # Now collapse any remaining extra non-spatial axes (if still >3D)
    # We assume the last two axes are spatial.
    if c.ndim > 3:
        # Merge all axes between spec and (y,x) into one by summing
        # Example: (n_spec, pol, y, x) -> sum over pol
        sum_axes = tuple(range(1, c.ndim - 2))
        c = np.nansum(c, axis=sum_axes)

    # Final sanity
    if c.ndim != 3:
        raise ValueError(f"After standardization, cube has shape {c.shape} (ndim={c.ndim}), expected 3D.")
    if c.shape[0] != int(n_spec):
        raise ValueError(f"After standardization, spectral length is {c.shape[0]}, expected {n_spec}.")

    return c, spec_axis

def velocity_axis_from_spectral_coord(
    spec_coord,
    spec_unit,
    *,
    line_value,
    line_unit,
    redshift,
    convention="radio",
):
    """
    Build a velocity axis (km/s) from a spectral coordinate axis that may be:
      - wavelength-like (convertible to meters)
      - frequency-like (convertible to Hz)
      - velocity-like (convertible to m/s)

    Parameters
    ----------
    spec_coord : Quantity array
        Axis values in native units (from header)
    spec_unit : Unit
        Native spectral unit
    line_value : float
        Rest wavelength OR rest frequency value provided by user
    line_unit : str or Unit
        Unit of line_value (e.g. "Angstrom", "um", "GHz")
    redshift : float
        Source redshift
    convention : {"radio","optical","relativistic"}
        Doppler convention when converting frequency to velocity.
        ALMA users usually want "radio".

    Returns
    -------
    vel_kms : np.ndarray
        Velocity axis in km/s *relative to the expected observed line position*.
        (You will still re-zero at systemic later, same as you do now.)
    spec_kind : str
        "wavelength" or "frequency" or "velocity"
    line_obs : Quantity
        Expected observed line coord (wavelength or frequency or velocity)
    """
    # parse user line input
    line_q = (line_value * u.Unit(line_unit)) if not isinstance(line_value, u.Quantity) else line_value.to(u.Unit(line_unit))

    # Case A: spectral axis is already velocity
    if spec_unit.is_equivalent(u.m/u.s):
        v_axis = spec_coord.to(u.km/u.s)
        # expected observed systemic velocity ~ 0 in many cubes; keep for consistency
        line_obs = (0.0 * u.km/u.s)
        vel_kms = (v_axis - line_obs).to_value(u.km/u.s)
        return vel_kms, "velocity", line_obs

    # Case B: wavelength axis
    if spec_unit.is_equivalent(u.m):
        wl = spec_coord.to(u.m)
        wl_rest = line_q.to(u.m)
        wl0 = wl_rest * (1.0 + redshift)  # observed wavelength
        vel = (c * (wl / wl0 - 1.0)).to(u.km/u.s)  # optical definition
        return vel.to_value(u.km/u.s), "wavelength", wl0

    # Case C: frequency axis (ALMA typical)
    if spec_unit.is_equivalent(u.Hz):
        nu = spec_coord.to(u.Hz)
        nu_rest = line_q.to(u.Hz)
        nu0 = nu_rest / (1.0 + redshift)  # observed frequency

        # Doppler conventions:
        # radio: v = c * (nu0 - nu) / nu0
        # optical: v = c * (nu0/nu - 1)
        # relativistic: use beta from ratio
        if convention.lower() == "radio":
            vel = (c * (nu0 - nu) / nu0).to(u.km/u.s)
        elif convention.lower() == "optical":
            vel = (c * (nu0 / nu - 1.0)).to(u.km/u.s)
        elif convention.lower() == "relativistic":
            r = (nu / nu0).to_value(u.one)
            beta = (1.0 - r**2) / (1.0 + r**2)
            vel = (beta * c).to(u.km/u.s)
        else:
            raise ValueError("convention must be 'radio', 'optical', or 'relativistic'")

        return vel.to_value(u.km/u.s), "frequency", nu0

    raise ValueError(
        f"Spectral axis unit '{spec_unit}' is not wavelength/frequency/velocity-like. "
        "Header likely does not store a usable spectral axis on axis 3."
    )

def spectral_axis_from_header_general(header, n_spec):
    """
    Read spectral axis linearly from CRVAL3/CRPIX3/CDELT3 (or CD3_3),
    and return (spec_coord, spec_unit).

    spec_coord is a Quantity array in the *native* unit from CUNIT3.
    """
    crval3 = float(header.get("CRVAL3"))
    crpix3 = float(header.get("CRPIX3"))
    cdelt3 = header.get("CD3_3", header.get("CDELT3"))
    if cdelt3 is None:
        raise KeyError("Could not find CD3_3 or CDELT3 for spectral axis in header.")
    cdelt3 = float(cdelt3)

    cunit3_raw = str(header.get("CUNIT3", "")).strip()
    cunit3_map = {
        "MICRON": "um",
        "micron": "um",
        "UM": "um",
        "DEGREE": "deg",
        "degree": "deg",
        "HZ": "Hz",
        "KHZ": "kHz",
        "MHZ": "MHz",
        "GHZ": "GHz",
    }
    cunit3 = cunit3_map.get(cunit3_raw, cunit3_raw)

    try:
        spec_unit = u.Unit(cunit3) if cunit3 else u.one
    except Exception:
        raise ValueError(f"Unrecognized spectral unit CUNIT3='{cunit3_raw}' after sanitizing -> '{cunit3}'")

    pix = (np.arange(n_spec, dtype=float) + 1.0)  # FITS is 1-based
    spec_coord = (crval3 + (pix - crpix3) * cdelt3) * spec_unit

    return spec_coord, spec_unit


def distances_from_z(z, cosmo):
    z = float(z)

    Dc = cosmo.comoving_distance(z).to(u.Mpc)
    Dl = cosmo.luminosity_distance(z).to(u.Mpc)
    Da = cosmo.angular_diameter_distance(z).to(u.Mpc)
    lookback = cosmo.lookback_time(z)

    # Robust: kpc per arcsec from angular diameter distance
    # 1 arcsec in radians:
    arcsec_in_rad = (1.0 * u.arcsec).to(u.rad).value
    scale = (Da.to(u.kpc).value) * arcsec_in_rad   # kpc / arcsec

    return Dc, Dl, Da, lookback, scale


def estimate_pa_from_mom1(
    vel_map,
    center_xy=None,
    smooth_sigma=1.0,
    mask=None,
    debug_plot=True,
    *,
    pixscale=1.0,
    nrebin=1,
    scale=None,
    xlimshow=None,
    ylimshow=None,
    n_boot=1000, 
    random_state=42, 
    return_samples=True,
    psf_sigma_arcsec=None,
    use_block_bootstrap=True,
    R_data_arcsec=None,
    R_data_err_arcsec = None,
    vel_range = None,
     ):
    """
    Estimate kinematic position angle (PA) from a velocity (moment-1) map.

    - PA is defined East of North (i.e., from North, clockwise).
    - The PCA gradient axis is rotated by +90 deg to get the major-axis PA.
    - Returns PA_deg in [0, 180).

    If debug_plot=True:
      - Shows the velocity map in ARCSEC with the star at (0,0),
      - Draws ONLY ONE arrow from the apex along the inferred PA direction,
      - Arrow length = 0.5 * (max |x| extent of the shown image),
      - The 180-deg ambiguity is resolved so that blue side is on the LEFT
        and red side on the RIGHT when looking along the arrow direction.
    """
    vel = np.array(vel_map, float)
    # ----- mask handling -----
    if mask is None:
        mask = np.isfinite(vel)
    else:
        mask = np.array(mask, bool) & np.isfinite(vel)
    if not np.any(mask):
        raise ValueError("No valid pixels found in velocity map.")

    # ----- smoothing (fill masked with median to avoid edge artefacts) -----
    fill = np.nanmedian(vel[mask])
    vel_smooth = gaussian_filter(np.where(mask, vel, fill), smooth_sigma)
    # ----- geometry / center in pixels -----
    ny, nx = vel.shape
    y, x = np.indices(vel.shape)
    if center_xy is None:
        x0, y0 = nx / 2.0, ny / 2.0
    else:
        x0, y0 = center_xy
    # ----- PCA inputs (pixel units are fine) -----
    Xp = (x - x0)[mask].ravel()
    Yp = (y - y0)[mask].ravel()
    V  = vel_smooth[mask].ravel()
    V  = V - np.nanmedian(V)
    coords = np.vstack([Xp, Yp]).T
    # weights from |V| (avoid zeros)
    w = np.abs(V)
    wmax = np.nanmax(w)
    if np.isfinite(wmax) and wmax > 0:
        w = w / wmax
    else:
        w = np.ones_like(w)

    # weighted covariance / principal axis
    mean = np.average(coords, axis=0, weights=w)
    coords_w = coords - mean
    cov = np.cov(coords_w.T, aweights=w)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]  # principal axis in pixel frame

    # ----- Convert to PA (East of North), then rotate +90° (major axis) -----
    # For a direction vector (dx, dy) in (x=East, y=North): PA = atan2(dx, dy)
    pa_grad_rad = np.arctan2(vec[0], vec[1])
    PA_deg_180 = (np.degrees(pa_grad_rad) + 90.0) % 180.0  # [0,180)

    PA_deg_360 = PA_deg_180  # candidate direction in [0,180) but interpreted as [0,360)

    pa_rad = np.deg2rad(PA_deg_360)
    uvec = np.array([np.sin(pa_rad), np.cos(pa_rad)])          # along PA (x,y)
    rvec = np.array([np.cos(pa_rad), -np.sin(pa_rad)])         # right-hand side normal (clockwise 90°)

    # classify pixels into right vs left of the axis passing through (x0,y0)
    # using dot(rvec, [Xp,Yp]) sign (Xp,Yp are centered pixel coords)
    side = (Xp * rvec[0] + Yp * rvec[1])

    right_sel = side > 0
    left_sel  = side < 0

    # robust means (weighted by |V| to focus on strong signal)
    def wmean(vals, ww):
        if vals.size == 0:
            return np.nan
        m = np.isfinite(vals) & np.isfinite(ww)
        if not np.any(m):
            return np.nan
        return np.average(vals[m], weights=ww[m])

    v_right = wmean(V[right_sel], w[right_sel])
    v_left  = wmean(V[left_sel],  w[left_sel])

    # If the convention is not met, flip by 180 deg
    # Target now: blue on LEFT, red on RIGHT => v_left < v_right
    if np.isfinite(v_right) and np.isfinite(v_left):
        if not (v_left < v_right):
            PA_deg_360 = (PA_deg_360 + 180.0) % 360.0
            pa_rad = np.deg2rad(PA_deg_360)
            uvec = np.array([np.sin(pa_rad), np.cos(pa_rad)])  # update oriented direction

    # ----- plotting in ARCSEC with star at (0,0) -----
    if debug_plot:
        arcsec_per_pix = float(pixscale) * float(nrebin)

        # coordinate extent in arcsec, centered at (x0,y0) => star is at (0,0)
        x_arc = (np.arange(nx) - x0) * arcsec_per_pix
        y_arc = (np.arange(ny) - y0) * arcsec_per_pix
        extent_arcsec = [x_arc[0], x_arc[-1], y_arc[0], y_arc[-1]]
        vmino = np.nanpercentile(vel[mask],1) if vel_range is None else vel_range[0]
        vmaxo = np.nanpercentile(vel[mask], 99) if vel_range is None else vel_range[1]

        plt.figure(figsize=(6.2, 5.4), dpi=150)
        im = plt.imshow(
            vel, origin='lower', cmap='RdBu_r',
            extent=extent_arcsec, interpolation='nearest',
            vmin=vmino,
            vmax=vmaxo
        )

        # Star at apex
        plt.scatter([0.0], [0.0], marker='*', s=300, c='k',
                    edgecolors='white', zorder=5)

        # Apply requested zoom FIRST (because arrow length depends on what you show)
        ax = plt.gca()
        if (xlimshow is not None) and (ylimshow is not None):
            ax.set_xlim(xlimshow[0], xlimshow[1])
            ax.set_ylim(ylimshow[0], ylimshow[1])

        # Arrow length = half the maximum |x| extent of the shown image
        xmin, xmax = ax.get_xlim()
        L = 0.5 * max(abs(xmin), abs(xmax))  # arcsec

        # Arrow endpoint in arcsec (x=East, y=North)
        dx = L * uvec[0]
        dy = L * uvec[1]

        # Draw ONLY ONE arrow from (0,0) to (dx,dy)
        ax.annotate(
            "", xy=(dx, dy), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.5, color="yellow"),
            zorder=6
        )

        # Optional: label
        ax.text(
            0.02, 0.98,
            f"PA = {PA_deg_360:.1f}°",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11,
            bbox=dict(facecolor="grey", alpha=0.7, edgecolor="none", pad=3), weight = 'bold' 
        )
        
            
            
        if (R_data_arcsec is not None) and np.isfinite(R_data_arcsec):
            ax.add_patch(Circle((0,0), R_data_arcsec, fill=False, lw=2, ls='--', ec='black', zorder=10))
            ax.text(0.02, 0.92, f"R={R_data_arcsec:.1f}\"",
                    transform=ax.transAxes, ha='left', va='top',
                    color='black', fontsize=12, zorder=11,
                    bbox=dict(facecolor="grey", alpha=0.7, edgecolor="none", pad=3), weight = 'bold' )
        
        if (R_data_err_arcsec is not None) and np.isfinite(R_data_err_arcsec):
            r1 = max(R_data_arcsec - R_data_err_arcsec, 0.0)
            r2 = R_data_arcsec + R_data_err_arcsec
            ax.add_patch(Circle((0,0), r1, fill=False, lw=1.4, ls='-', ec='grey', alpha=0.8, zorder=10))
            ax.add_patch(Circle((0,0), r2, fill=False, lw=1.4, ls='-', ec='grey', alpha=0.8, zorder=10))

        plt.colorbar(im, fraction=0.046, pad=0.02).set_label(r"$v_{\rm los}$ (km s$^{-1}$)")
        plt.xlabel(r"$\Delta$ RA ['']")
        plt.ylabel(r"$\Delta$ DEC ['']")
        plt.title("Estimated P.A. and Radius", fontsize = 14)

        # Optional top axis in kpc
        if scale is not None:
            def a2k(x): return x * float(scale)
            def k2a(x): return x / float(scale)
            secx = ax.secondary_xaxis('top', functions=(a2k, k2a))
            secx.set_xlabel('kpc')

        ax.set_aspect('equal')
        plt.tight_layout()
        #plt.show()
        
    
    PA_sigma = np.nan
    PA_boot = None
    
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(random_state)
    
        arcsec_per_pix = float(pixscale) * float(nrebin)
        psf_sigma_arcsec = psf_sigma_arcsec[0] if len(psf_sigma_arcsec)>=1 else psf_sigma_arcsec
    
        # Convert PSF sigma -> FWHM in pixels; set block size ~ FWHM
        if (use_block_bootstrap and psf_sigma_arcsec is not None
            and np.isfinite(psf_sigma_arcsec) and psf_sigma_arcsec > 0
            and arcsec_per_pix > 0):
            fwhm_pix = (2.355 * float(psf_sigma_arcsec)) / arcsec_per_pix
            block = int(np.clip(np.round(fwhm_pix), 3, 51))
        else:
            block = 1
    
        yy_all, xx_all = np.where(mask)
        N_all = yy_all.size
        if N_all < 20:
            # too few points for any meaningful uncertainty
            if return_samples:
                return round(PA_deg_360, 1), np.nan
            return round(PA_deg_360, 1), np.nan
    
        # Build an index map once: idx_map[y,x] = index in (yy_all,xx_all) or -1
        idx_map = np.full(mask.shape, -1, dtype=np.int32)
        idx_map[yy_all, xx_all] = np.arange(N_all, dtype=np.int32)
    
        def _pa_from_subset(sel_idx):
            Xp_b = (xx_all[sel_idx] - x0).astype(float)
            Yp_b = (yy_all[sel_idx] - y0).astype(float)
            V_b  = vel_smooth[yy_all[sel_idx], xx_all[sel_idx]].astype(float)
            V_b  = V_b - np.nanmedian(V_b)
    
            coords_b = np.vstack([Xp_b, Yp_b]).T
            w_b = np.abs(V_b)
            wmax_b = np.nanmax(w_b)
            w_b = (w_b / wmax_b) if (np.isfinite(wmax_b) and wmax_b > 0) else np.ones_like(w_b)
    
            # if all weights are ~0, skip
            if not np.isfinite(w_b).any() or np.nanmax(w_b) <= 0:
                return np.nan
    
            mean_b = np.average(coords_b, axis=0, weights=w_b)
            coords_w_b = coords_b - mean_b
            cov_b = np.cov(coords_w_b.T, aweights=w_b)
            if not np.all(np.isfinite(cov_b)):
                return np.nan
    
            eigvals_b, eigvecs_b = np.linalg.eigh(cov_b)
            vec_b = eigvecs_b[:, np.argmax(eigvals_b)]
    
            pa_grad_rad_b = np.arctan2(vec_b[0], vec_b[1])
            PA_b_180 = (np.degrees(pa_grad_rad_b) + 90.0) % 180.0
    
            # same orientation rule as your original code
            pa_rad_b = np.deg2rad(PA_b_180)
            rvec_b = np.array([np.cos(pa_rad_b), -np.sin(pa_rad_b)])
            side_b = (Xp_b * rvec_b[0] + Yp_b * rvec_b[1])
    
            right_sel_b = side_b > 0
            left_sel_b  = side_b < 0
    
            def wmean(vals, ww):
                m = np.isfinite(vals) & np.isfinite(ww)
                if not np.any(m):
                    return np.nan
                return np.average(vals[m], weights=ww[m])
    
            v_right_b = wmean(V_b[right_sel_b], w_b[right_sel_b])
            v_left_b  = wmean(V_b[left_sel_b],  w_b[left_sel_b])
    
            PA_b_360 = float(PA_b_180)
            if np.isfinite(v_right_b) and np.isfinite(v_left_b):
                if not (v_left_b < v_right_b):
                    PA_b_360 = (PA_b_360 + 180.0) % 360.0
    
            return PA_b_360 % 180.0
    
        PA_list = []
    
        if block <= 1:
            # classic bootstrap on pixels
            for _ in range(int(n_boot)):
                sel = rng.integers(0, N_all, size=N_all)
                pa_b = _pa_from_subset(sel)
                if np.isfinite(pa_b):
                    PA_list.append(pa_b)
        else:
            ny0, nx0 = mask.shape
    
            # choose a reasonable number of blocks per bootstrap
            # (valid area / block area, with caps)
            n_blocks = int(np.clip(np.ceil(N_all / (block * block)), 8, 250))
    
            # try multiple attempts per bootstrap so we don't end up empty
            max_attempts = 10
    
            for _ in range(int(n_boot)):
                sel_idx_parts = []
    
                for _attempt in range(max_attempts):
                    ys = rng.integers(0, max(1, ny0 - block + 1), size=n_blocks)
                    xs = rng.integers(0, max(1, nx0 - block + 1), size=n_blocks)
    
                    sel_idx_parts.clear()
                    for yb, xb in zip(ys, xs):
                        block_idx = idx_map[yb:yb+block, xb:xb+block]
                        good = block_idx[block_idx >= 0]
                        if good.size:
                            sel_idx_parts.append(good)
    
                    if sel_idx_parts:
                        sel_idx = np.unique(np.concatenate(sel_idx_parts))
                        # relax threshold: small maps may only have ~20-40 pixels
                        if sel_idx.size >= 20:
                            pa_b = _pa_from_subset(sel_idx)
                            if np.isfinite(pa_b):
                                PA_list.append(pa_b)
                            break  # success -> next bootstrap
    
        # compute sigma
        if len(PA_list) >= 10:
            PA_boot = np.asarray(PA_list, dtype=float)
            _, PA_sigma = pa180_circ_mean_std(PA_boot)
        else:
            PA_sigma = np.nan
    
    PA_deg_180 = PA_deg_360 % 180.0  # keep consistent after orientation flip

    if return_samples:
        return round(PA_deg_360, 1), round(PA_sigma, 1)
    else:
        return round(PA_deg_360, 1), round(PA_sigma, 1)
    

    
def estimate_radius_from_encircled_flux_with_uncertainty(
    flux_map,
    center_xy_pix,
    pixscale_arcsec,
    nrebin,
    psf_sigma_arcsec,
    frac=0.98,
    use_positive_only=True,
    n_boot=500,
    rng_seed=1,
):
    """
    Returns:
      R_obs_arcsec, R_int_arcsec, R_int_err_arcsec, boot_R_int_arcsec (array)

    Uncertainty is bootstrap scatter (16-84% / 2) on the PSF-corrected radius.
    """
    f = np.array(flux_map, float)
    ny, nx = f.shape
    x0, y0 = float(center_xy_pix[0]), float(center_xy_pix[1])

    yy, xx = np.indices((ny, nx))
    rr_pix = np.hypot(xx - x0, yy - y0)

    good = np.isfinite(f)
    if use_positive_only:
        good &= (f > 0)

    if not np.any(good):
        return np.nan, np.nan, np.nan, np.array([])

    r = rr_pix[good].ravel()
    w = f[good].ravel()
    tot = np.sum(w)
    if not np.isfinite(tot) or tot <= 0:
        return np.nan, np.nan, np.nan, np.array([])

    pix_arcsec = float(pixscale_arcsec) * float(nrebin)
    fwhm_arcsec = 2.355 * float(psf_sigma_arcsec[0])

    def _encircled_radius_arcsec(r_pix, w_):
        # sort by radius
        idx = np.argsort(r_pix)
        r_sorted = r_pix[idx]
        w_sorted = w_[idx]
        tot_ = np.sum(w_sorted)
        if tot_ <= 0:
            return np.nan
        cfrac = np.cumsum(w_sorted) / tot_
        R_pix = np.interp(float(frac), cfrac, r_sorted)
        return float(R_pix * pix_arcsec)

    # point estimate
    R_obs_arcsec = _encircled_radius_arcsec(r, w)
    R_int_arcsec = float(np.sqrt(max(R_obs_arcsec**2 - (0.5 * fwhm_arcsec)**2, 0.0)))

    # bootstrap
    rng = np.random.default_rng(rng_seed)
    n = r.size
    boot = np.full(int(n_boot), np.nan, float)

    for k in range(int(n_boot)):
        ii = rng.integers(0, n, size=n)  # resample pixels with replacement
        Rb_obs = _encircled_radius_arcsec(r[ii], w[ii])
        if np.isfinite(Rb_obs):
            boot[k] = np.sqrt(max(Rb_obs**2 - (0.5 * fwhm_arcsec)**2, 0.0))

    boot = boot[np.isfinite(boot)]
    if boot.size < 20:
        R_int_err = np.nan
    else:
        p16, p84 = np.percentile(boot, [16, 84])
        R_int_err = 0.5 * (p84 - p16)

    return R_obs_arcsec, R_int_arcsec, R_int_err, boot


def plot_kin_maps_3x3(
    obs, m,
    xy_AGN=None,
    xrange = None,
    yrange = None,
    fl_percent=(5, 99),
    vrange=None,
    sigrange=None,
    resid_ranges=(0.1, 20, 20),
    cmap_flux='inferno',
    cmap_vel='RdBu_r',
    cmap_sig='viridis',
    nticks=5,
    cbar_pad=0.0025,
    cbar_w=0.012,
    cbar_gap=0.012,
    cbar_vpad=0.015,
    # --- PSF overlay on DATA row (row=0) ---
    psf_bmaj=None,          # major axis size (arcsec)
    psf_bmin=None,          # minor axis size (arcsec)
    psf_pa=0,             # position angle of PSF ellipse (deg, CCW from +x in plot coords)
    psf_loc='lower left',   # 'lower left' or 'lower right' etc.
    psf_pad_frac=0.1,      # padding from edges (fraction of axis span)
    psf_line_angle=45.0,    # internal line inclination (deg, CCW from +x)
    psf_nlines=9,           # number of internal lines
    psf_color='black',
    psf_lw=1.6,
    psf_line_lw=1.0
):
    """
    3x3 panel plot:
      Row 0: DATA  (flux, vel, sig)  + optional PSF ellipse marker in each DATA subpanel
      Row 1: MODEL (flux, vel, sig)  (MODEL pixels that are NaN or 0 are shown white)
      Row 2: RESID (data-model)

    Notes:
    - Coordinates are shifted so AGN is at (0,0): axes are true ΔRA/ΔDec.
    - Ticks are automatic and guaranteed to include 0.
    - One tall colorbar per column spanning DATA+MODEL + separate residual colorbar.
    - MODEL pixels that are NaN or exactly 0 are rendered white.
    - If psf_bmaj and psf_bmin are provided (arcsec), a PSF ellipse with inclined internal
      lines is drawn in the bottom-left (or chosen corner) of each DATA row panel.
    """
    
    extent = [m.cube['xextent'][0], m.cube['xextent'][1], m.cube['yextent'][0], m.cube['yextent'][1]]
    psf_bmaj *= 2.355    # arcsec
    psf_bmin *= 2.355    # arcsec
   

    # ---------- maps ----------
    flu  = np.array(obs.maps['flux'], float)
    ve   = np.array(obs.maps['vel'],  float)
    si   = np.array(obs.maps['sig'],  float)

    fluxx = np.array(m.maps['flux'], float)
    vell  = np.array(m.maps['vel'],  float)
    sigg  = np.array(m.maps['sig'],  float)

    # ----------------------------------------------------
    # MODEL and DATA pixels that are NaN OR zero -> NaN (so they plot white)
    # ----------------------------------------------------
    flu[(~np.isfinite(flu)) | (flu <= 0)] = np.nan
    ve[(~np.isfinite(ve))   | (ve  == 0)] = np.nan
    si[(~np.isfinite(si))   | (si  == 0)] = np.nan

    fluxx[(~np.isfinite(fluxx)) | (fluxx <= 0)] = np.nan
    vell[(~np.isfinite(vell))   | (vell  == 0)] = np.nan
    sigg[(~np.isfinite(sigg))   | (sigg  == 0)] = np.nan

    # ---------- extent ----------
    if extent is None:
        extent = [m.cube['xextent'][0], m.cube['xextent'][1],
                  m.cube['yextent'][0], m.cube['yextent'][1]]
    xmin, xmax, ymin, ymax = extent

    # ---------- shift to AGN-centered coordinates ----------
    if xy_AGN is not None:
        x0, y0 = float(xy_AGN[0]), float(xy_AGN[1])
    else:
        x0, y0 = 0.0, 0.0
    extent0 = [xmin - x0, xmax - x0, ymin - y0, ymax - y0]

    # ---------- colormaps ----------
    cmap_flux_obj = cm.get_cmap(cmap_flux).copy()
    cmap_vel_obj  = cm.get_cmap(cmap_vel).copy()
    cmap_sig_obj  = cm.get_cmap(cmap_sig).copy()
    for _cmap in (cmap_flux_obj, cmap_vel_obj, cmap_sig_obj):
        _cmap.set_bad(color='white')  # NaNs shown white

    # ---------- log flux ----------
    pos = flu[np.isfinite(flu) & (flu > 0)]
    floor = np.nanpercentile(pos, 1) if pos.size else 1e-30
    flu_log   = np.log10(np.clip(flu,  floor, None))
    fluxx_log = np.log10(np.clip(fluxx, floor, None))

    # ---------- residuals ----------
    res_fl_map = flu_log - fluxx_log
    res_v_map  = ve - vell
    res_s_map  = si - sigg

    # ---------- ranges ----------
    if vrange is None:
        vv = ve[np.isfinite(ve)]
        vrange = np.nanpercentile(vv, [1, 99]) if vv.size else (-1, 1)

    if sigrange is None:
        ss = si[np.isfinite(si)]
        sigrange = np.nanpercentile(ss, [1, 99]) if ss.size else (0, 1)

    ff = flu_log[np.isfinite(flu_log)]
    fl_vmin, fl_vmax = np.nanpercentile(ff, fl_percent) if ff.size else (0, 1)

    flrange_r  = (-abs(resid_ranges[0]), abs(resid_ranges[0]))
    vrange_r   = (-abs(resid_ranges[1]), abs(resid_ranges[1]))
    sigrange_r = (-abs(resid_ranges[2]), abs(resid_ranges[2]))

    # ---------- figure ----------
    fig, ax = plt.subplots(
        3, 3, sharex=True, sharey=True,
        subplot_kw={'aspect': 'equal'},
        figsize=(18, 12), dpi=250
    )
    plt.subplots_adjust(hspace=0.0, wspace=0.25)

    ims = [[None]*3 for _ in range(3)]

    # ---------- draw ----------
    # Flux column
    ims[0][0] = ax[0][0].imshow(flu_log, origin='lower', extent=extent0,
                                cmap=cmap_flux_obj, vmin=fl_vmin, vmax=fl_vmax, interpolation='nearest')
    ims[1][0] = ax[1][0].imshow(fluxx_log, origin='lower', extent=extent0,
                                cmap=cmap_flux_obj, vmin=fl_vmin, vmax=fl_vmax, interpolation='nearest')
    ims[2][0] = ax[2][0].imshow(res_fl_map, origin='lower', extent=extent0,
                                cmap=cmap_flux_obj, vmin=flrange_r[0], vmax=flrange_r[1], interpolation='nearest')

    # Velocity column
    ims[0][1] = ax[0][1].imshow(ve, origin='lower', extent=extent0,
                                cmap=cmap_vel_obj, vmin=vrange[0], vmax=vrange[1], interpolation='nearest')
    ims[1][1] = ax[1][1].imshow(vell, origin='lower', extent=extent0,
                                cmap=cmap_vel_obj, vmin=vrange[0], vmax=vrange[1], interpolation='nearest')
    ims[2][1] = ax[2][1].imshow(res_v_map, origin='lower', extent=extent0,
                                cmap=cmap_vel_obj, vmin=vrange_r[0], vmax=vrange_r[1], interpolation='nearest')

    # Sigma column
    ims[0][2] = ax[0][2].imshow(si, origin='lower', extent=extent0,
                                cmap=cmap_sig_obj, vmin=sigrange[0], vmax=sigrange[1], interpolation='nearest')
    ims[1][2] = ax[1][2].imshow(sigg, origin='lower', extent=extent0,
                                cmap=cmap_sig_obj, vmin=sigrange[0], vmax=sigrange[1], interpolation='nearest')
    ims[2][2] = ax[2][2].imshow(res_s_map, origin='lower', extent=extent0,
                                cmap=cmap_sig_obj, vmin=sigrange_r[0], vmax=sigrange_r[1], interpolation='nearest')

    # use user ranges if provided (already AGN-centered coordinates)
    if xrange is not None:
        if len(xrange) != 2:
            raise ValueError("xrange must be [xmin, xmax]")
        xlim_use = (float(xrange[0]), float(xrange[1]))
    else:
        xlim_use = (extent0[0], extent0[1])
    
    if yrange is not None:
        if len(yrange) != 2:
            raise ValueError("yrange must be [ymin, ymax]")
        ylim_use = (float(yrange[0]), float(yrange[1]))
    else:
        ylim_use = (extent0[2], extent0[3])
    
    for i in range(3):
        for j in range(3):
            ax[i][j].set_xlim(*xlim_use)
            ax[i][j].set_ylim(*ylim_use)

    # ---------- AGN marker at (0,0) ----------
    if xy_AGN is not None:
        for i in range(3):
            for j in range(3):
                ax[i][j].scatter(0.0, 0.0, s=200, marker='*', color='black', zorder=6)

    # ---------- titles/labels ----------
    ax[0][0].set_title(r'Log Flux', fontsize=30)
    ax[0][1].set_title(r'Mom-1 (km s$^{-1}$)', fontsize=30)
    ax[0][2].set_title(r'Mom-2 (km s$^{-1}$)', fontsize=30)

    ax[0][0].set_ylabel("DATA\n" + r"$\Delta$ Dec ['']", fontsize=28)
    ax[1][0].set_ylabel("MODEL\n" + r"$\Delta$ Dec ['']", fontsize=28)
    ax[2][0].set_ylabel("RESIDUAL\n" + r"$\Delta$ Dec ['']", fontsize=28)

    for j in range(3):
        ax[2][j].set_xlabel(r"$\Delta$ RA ['']", fontsize=28)

    # ---------- ticks: automatic + include 0 + no scientific notation ----------
    def _apply_ticks(a):
        a.xaxis.set_major_locator(MaxNLocator(nbins=nticks, symmetric=True))
        a.yaxis.set_major_locator(MaxNLocator(nbins=nticks, symmetric=True))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        a.xaxis.set_major_formatter(fmt)
        a.yaxis.set_major_formatter(fmt)
        a.tick_params(direction='in', labelsize=22, width=2)

    for i in range(3):
        for j in range(3):
            _apply_ticks(ax[i][j])

    # ---------- NEW: PSF ellipse with inclined internal lines on DATA row ----------
    def _corner_center(xmin_, xmax_, ymin_, ymax_, loc, pad_frac):
        xr = xmax_ - xmin_
        yr = ymax_ - ymin_
        pad_x = pad_frac * xr
        pad_y = pad_frac * yr

        loc = str(loc).lower().replace('-', ' ')
        if loc in ("lower left", "bottom left"):
            return xmin_ + pad_x, ymin_ + pad_y
        if loc in ("lower right", "bottom right"):
            return xmax_ - pad_x, ymin_ + pad_y
        if loc in ("upper left", "top left"):
            return xmin_ + pad_x, ymax_ - pad_y
        if loc in ("upper right", "top right"):
            return xmax_ - pad_x, ymax_ - pad_y
        # fallback
        return xmin_ + pad_x, ymin_ + pad_y

    def _draw_psf(ax_, bmaj, bmin, pa_deg, loc, pad_frac, line_angle_deg, nlines):
        # position for the ellipse center (in plot coords = arcsec offsets)
        xmin_, xmax_ = ax_.get_xlim()
        ymin_, ymax_ = ax_.get_ylim()
        cx, cy = _corner_center(xmin_, xmax_, ymin_, ymax_, loc, pad_frac)
        ell = Ellipse(
            (cx, cy),
            width=bmaj,
            height=bmin,
            angle=pa_deg,
            facecolor='white',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            zorder=6
                )
        ax_.add_patch(ell)

        # internal inclined lines, clipped to ellipse
        # build parallel lines covering a bounding box around the ellipse
        theta = np.deg2rad(line_angle_deg)
        # direction along the line
        dx, dy = np.cos(theta), np.sin(theta)
        # normal to the line
        nx, ny = -dy, dx

        # span along normal direction (make sure it covers ellipse fully)
        # use half-diagonal as safe bound
        half_diag = 0.7 * np.hypot(bmaj, bmin)

        # offsets along normal for each line
        offs = np.linspace(-half_diag, half_diag, nlines)

        # line half-length (big enough)
        L = 2.0 * half_diag
        for o in offs:
            # a point on the line, shifted by o along the normal
            px = cx + o * nx
            py = cy + o * ny
            x0l = px - L * dx
            y0l = py - L * dy
            x1l = px + L * dx
            y1l = py + L * dy

            ln, = ax_.plot([x0l, x1l], [y0l, y1l],
                           color=psf_color, lw=psf_line_lw, zorder=7)
            ln.set_clip_path(ell)

    if (psf_bmaj is not None) and (psf_bmin is not None):
        try:
            bmaj = float(psf_bmaj)
            bmin = float(psf_bmin)
            if (bmaj > 0) and (bmin > 0):
                for i in range(3):
                    for j in range(3):  # DATA row only
                        _draw_psf(ax[i][j], bmaj, bmin, float(psf_pa),
                                  psf_loc, float(psf_pad_frac),
                                  float(psf_line_angle), int(psf_nlines))
        except Exception:
            # If something is wrong with inputs, just skip PSF overlay
            pass

    # ---------- colorbars ----------
    for j in range(3):
        p_top = ax[0][j].get_position()
        p_mid = ax[1][j].get_position()
        p_bot = ax[2][j].get_position()

        x_cbar = p_top.x1 + cbar_pad

        y0_main = p_mid.y0 + cbar_vpad
        y1_main = p_top.y1 - cbar_vpad
        y0_res  = p_bot.y0 + cbar_vpad
        y1_res  = p_bot.y1 - cbar_vpad

        if (y0_main - y1_res) < cbar_gap:
            shift = 0.5 * (cbar_gap - (y0_main - y1_res))
            y0_main += shift
            y1_res  -= shift

        cax_main = fig.add_axes([x_cbar, y0_main, cbar_w, y1_main - y0_main])
        fig.colorbar(ims[0][j], cax=cax_main).ax.tick_params(labelsize=22)

        cax_res = fig.add_axes([x_cbar, y0_res, cbar_w, y1_res - y0_res])
        fig.colorbar(ims[2][j], cax=cax_res).ax.tick_params(labelsize=22)

    #plt.show()





################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
#%% functions to built the model

def _as_shell_ranges(r_range, n_shells):
    """Return radius_range formatted for model:
       - if n_shells == 1: returns [r0, r1] (unchanged)
       - if n_shells > 1 : returns [[r0,r1],[r1,r2],...]
    """
    if n_shells <= 1:
        return r_range
    edges = np.linspace(r_range[0], r_range[1], n_shells + 1)
    return [[edges[i], edges[i + 1]] for i in range(n_shells)]


def _generate_by_geometry(model, geometry, fluxpars, v):
    """:
       cylindrical -> vel3=v
       spherical   -> vel1=v
    """
    if geometry == "cylindrical":
        model.generate_clouds(flux_pars=fluxpars, vel1_pars=[0], vel2_pars=[0], vel3_pars=[v])
    else:
        model.generate_clouds(flux_pars=fluxpars, vel1_pars=[v], vel2_pars=[0], vel3_pars=[0])


def _make_single_km_component(
    *,
    npt, 
    geometry, radius_range, theta_range, phi_range, zeta_range,
    logradius, flux_func, vel1_func, vel2_func, vel3_func,
    vel_sigma, psf_sigma, lsf_sigma,
    cube_range, cube_nbins,
    fluxpars,
    v,                      
    xycenter, alpha, beta, gamma, vsys
):
    m = model(
        npt=npt, 
        geometry=geometry,
        radius_range=radius_range, theta_range=theta_range, phi_range=phi_range, zeta_range=zeta_range,
        logradius=logradius,
        flux_func=flux_func,
        vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
        vel_sigma=vel_sigma,
        psf_sigma=psf_sigma, lsf_sigma=lsf_sigma,
        cube_range=cube_range, cube_nbins=cube_nbins
    )
    _generate_by_geometry(m, geometry, fluxpars, v)
    m.observe_clouds(xycenter=xycenter, alpha=alpha, beta=beta, gamma=gamma, vsys=vsys)
    return m


def _make_multishell_component(
    *,
    npt_total, n_shells,
    geometry, radius_range_shells, theta_range, phi_range, zeta_range,
    logradius, flux_func, vel1_func, vel2_func, vel3_func,
    vel_sigma, psf_sigma, lsf_sigma,
    cube_range, cube_nbins,
    fluxpars,
    v_arr, beta_arr,          # arrays length n_shells
    xycenter, alpha, gamma, vsys
):
    # Build first shell as base model
    npt_shell = int(npt_total / n_shells)

    m0 = _make_single_km_component(
        npt=npt_shell,
        geometry=geometry, radius_range=radius_range_shells[0],
        theta_range=theta_range, phi_range=phi_range, zeta_range=zeta_range,
        logradius=logradius, flux_func=flux_func,
        vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
        vel_sigma=vel_sigma, psf_sigma=psf_sigma, lsf_sigma=lsf_sigma,
        cube_range=cube_range, cube_nbins=cube_nbins,
        fluxpars=fluxpars,
        v=v_arr[0],
        xycenter=xycenter, alpha=alpha, beta=beta_arr[0], gamma=gamma, vsys=vsys
    )

    # Add remaining shells
    for i in range(1, n_shells):
        mi = _make_single_km_component(
            npt=npt_shell, 
            geometry=geometry, radius_range=radius_range_shells[i],
            theta_range=theta_range, phi_range=phi_range, zeta_range=zeta_range,
            logradius=logradius, flux_func=flux_func,
            vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
            vel_sigma=vel_sigma, psf_sigma=psf_sigma, lsf_sigma=lsf_sigma,
            cube_range=cube_range, cube_nbins=cube_nbins,
            fluxpars=fluxpars,
            v=v_arr[i],
            xycenter=xycenter, alpha=alpha, beta=beta_arr[i], gamma=gamma, vsys=vsys
        )
        m0.add_model(mi)

    return m0

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
#%% fitting functions

_FIT_CTX = {}

def set_fit_context(**kwargs):
    """
    Store fit/runtime parameters that were previously global variables in your notebook/script.

    Call this ONCE in your main code after you define obs, vel axis, origin, etc.

    Example:
        set_fit_context(
            geometry=geometry,
            FIT_MODE=FIT_MODE,
            obs=obs,
            vel_axis=vel.value,
            origin=origin,
            pixscale= pixscale,
            xy_AGN=xy_AGN,
            gamma_model=gamma_model,
            num_shells=num_shells,
            rin_pix=rin_pix, rout_pix=rout_pix,
            aperture=aperture, double_cone=double_cone,
            SIGMA_PERC_KMS=SIGMA_PERC_KMS,
            perc=perc,
            perc_weights=perc_weights,
            loss=loss,
            CRPS_QGRID=CRPS_QGRID,
            scale=scale,
            RT_ARCSEC=RT_ARCSEC,
            npt=npt,
            use_seeds=use_seeds,
            seeds=seeds,
            radius_range_model=radius_range_model,
            theta_range=theta_range,
            phi_range=phi_range,
            zeta_range=zeta_range,
            logradius=logradius,
            psf_sigma=psf_sigma,
            lsf_sigma=lsf_sigma,
            vel_sigma=vel_sigma,
        )
    """
    _FIT_CTX.update(kwargs)

def _ctx_get(key, default=None, *, required=False):
    if key in _FIT_CTX:
        return _FIT_CTX[key]
    if required:
        raise KeyError(f"km context missing required key: '{key}'. "
                       f"Did you forget set_fit_context({key}=...)?")
    return default


def make_cone_spatial_mask(shape_yx, center_xy, pa_deg, opening_deg,
                           mode="bicone", axis_sign=+1):
    """
    Spatial cone mask in the sky plane.

    PA convention (astronomical):
      - pa_deg=0   : points North (up, +y)
      - pa_deg=90  : points East  (right, +x)

    opening_deg is FULL opening angle (half-angle = opening_deg/2).

    mode:
      - "bicone": symmetric, includes both +axis and -axis directions
      - "single": includes only one side; axis_sign=+1 keeps +axis, -1 keeps -axis
    """
    ny, nx = shape_yx
    x0, y0 = float(center_xy[0]), float(center_xy[1])

    yy, xx = np.indices((ny, nx))
    dx = xx - x0
    dy = yy - y0

    # axis unit vector from PA (0=N => (ax,ay)=(0,1); 90=E => (1,0))
    pa = np.deg2rad(float(pa_deg))
    ax = np.sin(pa)
    ay = np.cos(pa)

    r = np.hypot(dx, dy)
    # avoid division by zero at center pixel
    r = np.where(r == 0, 1.0, r)
    ux = dx / r
    uy = dy / r

    # cos(angle) between pixel direction and axis direction
    dot = ux * ax + uy * ay

    half = 0.5 * float(opening_deg)
    cmin = np.cos(np.deg2rad(half))

    mode = str(mode).lower()
    if mode in ("bicone", "double", "double_cone", "bi", "bi-cone"):
        # symmetric: accept both signs
        inside = np.abs(dot) >= cmin
    elif mode in ("single", "cone"):
        inside = (dot * float(axis_sign)) >= cmin
    else:
        raise ValueError("mode must be 'single' or 'bicone'")

    return inside


def apply_spatial_mask_to_cube(cube_spec_yx, mask_yx, mode="zero"):
    """
    mode:
      - "zero": set masked pixels to 0
      - "nan": set masked pixels to NaN
    """
    out = np.array(cube_spec_yx, copy=True)
    if mode == "zero":
        out[:, mask_yx] = 0.0
    elif mode == "nan":
        out[:, mask_yx] = np.nan
    else:
        raise ValueError("mode must be 'zero' or 'nan'")
    return out



def _rotate_to_pa(x, y, pa_deg):
    th = np.radians(pa_deg)
    xr =  x*np.cos(th) + y*np.sin(th)
    yr = -x*np.sin(th) + y*np.cos(th)
    return xr, yr

def pa_astro_to_mask_angle(pa_deg, geometry):
   
    pa = float(pa_deg) % 360.0
    g = geometry.lower()
    if g == "cylindrical":
        pa_major_astro = (pa - 90.0) % 360.0   # convert minor->major
    else:
        pa_major_astro = pa                    # already major (cone axis)
    return pa_astro_to_math(pa_major_astro)    # astro -> math angle for _rotate_to_pa

def pa_astro_to_math(pa_deg):
    """Convert astro PA (East of North; from +Y clockwise) to math/mpl angle (from +X CCW)."""
    return (90.0 - float(pa_deg)) % 360.0


def _finite_mask_from_cube(cube):
    plane = np.nansum(cube, axis=0)
    return np.isfinite(plane) & (plane > 0)

def _pick_mask(mode, obs_mask, model_mask):
    modes = {
        "obs":          obs_mask,
        "model":        model_mask,
        "union":        (obs_mask | model_mask),
        "intersection": (obs_mask & model_mask),
    }
    return modes.get(mode, obs_mask)

def _elliptical_radius_and_intrinsic_angle(shape, center_xy, inc_deg, pa_deg, *, use_cos=False):
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    xx = xx - center_xy[0]
    yy = yy - center_xy[1]

    if use_cos:
        q = np.clip(np.cos(np.radians(inc_deg)), 0.01, 1.0)
    else:
        q = np.clip(np.sin(np.radians(inc_deg)), 0.01, 1.0)

    xr, yr = _rotate_to_pa(xx, yy, pa_deg)
    rell = np.sqrt(xr**2 + (yr / q)**2)
    ang_int = np.arctan2((yr / q), xr)
    return rell, ang_int, q

def plot_bestfit_summary(best, r_edges_pix, arcsec_per_pix, scale):

    # ---- helpers for secondary axis ----
    def a2k(x):
        return np.asarray(x, dtype=float) * float(scale)

    def k2a(x):
        return np.asarray(x, dtype=float) / float(scale)

    # ---- radii at shell midpoints ----
    r_edges_pix = np.asarray(r_edges_pix, dtype=float)
    if r_edges_pix.ndim != 1 or r_edges_pix.size < 2:
        raise ValueError("r_edges_pix must be a 1D array of length (n_shells+1).")

    r_mid_pix = 0.5 * (r_edges_pix[:-1] + r_edges_pix[1:])
    r_arcsec = r_mid_pix * float(arcsec_per_pix)

    # ---- pull best-fit arrays ----
    beta = np.asarray(best.get("beta", None), dtype=float)
    v    = np.asarray(best.get("v", None), dtype=float)
    if beta.size == 0 or v.size == 0:
        raise ValueError("best must contain non-empty 'beta' and 'v' arrays.")

    beta_err = np.asarray(best.get("beta_err", np.full_like(beta, np.nan)), dtype=float)
    v_err    = np.asarray(best.get("v_err",    np.full_like(v,    np.nan)), dtype=float)

    # make lengths consistent (trim to smallest)
    n = int(min(r_arcsec.size, beta.size, v.size, beta_err.size, v_err.size))
    r_arcsec  = r_arcsec[:n]
    beta      = beta[:n]
    v         = v[:n]
    beta_err  = beta_err[:n]
    v_err     = v_err[:n]

    # ---- mask finite ----
    ok_beta = np.isfinite(r_arcsec) & np.isfinite(beta)
    ok_v    = np.isfinite(r_arcsec) & np.isfinite(v)

    # ---- plotting ----
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    fig.patch.set_facecolor("white")

    def prettify(ax, top_label):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", labelsize=10, length=6, width=1)
        ax.tick_params(axis="both", which="minor", labelsize=8, length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", alpha=0.20)

        secx = ax.secondary_xaxis("top", functions=(a2k, k2a))
        secx.set_xlabel(top_label, fontsize=12, labelpad=8)
        secx.tick_params(axis="x", which="major", labelsize=10, length=5, width=1)
        secx.tick_params(axis="x", which="minor", labelsize=8, length=3, width=0.8)

    # ---- beta panel ----
    ax = axes[0]
    ax.errorbar(
        r_arcsec[ok_beta], beta[ok_beta],
        yerr=np.where(np.isfinite(beta_err[ok_beta]), beta_err[ok_beta], 0.0),
        fmt="o", ms=5, mfc="none", mec="blue", mew=1.2,
        ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2, zorder=3
    )
    ax.set_xlabel("Radius (arcsec)", fontsize=12)
    ax.set_ylabel(r"$\beta$ (deg)", fontsize=12)
    ax.set_title("Best-fit Inclination", fontsize=16, pad=10)
    prettify(ax, "Radius (kpc)")

    # ---- velocity panel ----
    ax = axes[1]
    ax.errorbar(
        r_arcsec[ok_v], v[ok_v],
        yerr=np.where(np.isfinite(v_err[ok_v]), v_err[ok_v], 0.0),
        fmt="o", ms=5, mfc="none", mec="blue", mew=1.2,
        ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2, zorder=3
    )
    ax.set_xlabel("Radius (arcsec)", fontsize=12)
    ax.set_ylabel(r"$v$ (km s$^{-1}$)", fontsize=12)
    ax.set_title("Best-fit Velocity", fontsize=16, pad=10)
    prettify(ax, "Radius (kpc)")

    plt.tight_layout()
    #plt.show()



def build_v_grid_and_label(geometry: str, fit_mode: str):
    geom = str(geometry).lower()
    mode = str(fit_mode)

    v_min  = float(_ctx_get("v_min", required=True))
    v_max  = float(_ctx_get("v_max", required=True))
    step_v = float(_ctx_get("step_v", required=True))
    n_geom = int(_ctx_get("n_geom_v", 50))

    if geom == "cylindrical" and mode == "disk_kepler":
        v_arr = np.geomspace(v_min, v_max, n_geom)
        return v_arr, None, r"$M_\bullet$ ($M_\odot$)"

    if geom == "cylindrical" and mode == "NSC":
        v_arr = np.geomspace(v_min, v_max, n_geom)
        return v_arr, None, r"$A$"

    if geom == "cylindrical" and mode == "Plummer":
        v_arr = np.geomspace(v_min, v_max, n_geom)
        return v_arr, None, r"$M_0$ ($M_\odot$)"

    if geom == "cylindrical" and mode == "disk_arctan":
        v_arr = np.arange(v_min, v_max + step_v, step_v)
        return v_arr, None, r"$V_{\max}$ (km s$^{-1}$)"

    v_arr = np.arange(v_min, v_max + step_v, step_v)
    return v_arr, None, r"$v$ (km s$^{-1}$)"





def plot_residual_maps_cone(chi_squared_map, beta_array, v_array, num_shells, *,
                            best=None, cmap='inferno', y_label=r"$v$ (km s$^{-1}$)"):

    chi_squared_map = np.asarray(chi_squared_map)
    num_shells = int(min(int(num_shells), chi_squared_map.shape[0]))
    ncols = min(4, int(num_shells))
    nrows = int(math.ceil(num_shells / ncols))
    fig_w = 3.2 * ncols
    fig_h = 3.0 * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h), dpi=140,
        sharex=True, sharey=True,
        squeeze=False
    )
    axes_flat = axes.ravel()

    beta_array = np.asarray(beta_array, float)
    v_array    = np.asarray(v_array, float)

    beta_min, beta_max = float(np.nanmin(beta_array)), float(np.nanmax(beta_array))
    v_min, v_max       = float(np.nanmin(v_array)),   float(np.nanmax(v_array))

    for i in range(int(num_shells)):
        
        ax = axes_flat[i]
        chi2 = np.asarray(chi_squared_map[i], float)

        if not np.any(np.isfinite(chi2)):
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_title(f"Shell {i+1}")
            continue

        # log10(chi2) but only where > 0
        Z = np.full_like(chi2, np.nan, dtype=float)
        pos = np.isfinite(chi2) & (chi2 > 0)
        Z[pos] = np.log10(chi2[pos])

        if np.any(pos):
            lo = np.nanpercentile(Z[pos], 1)
            hi = np.nanpercentile(Z[pos], 99)
            # keep sensible bounds
            vmin = lo * 0.75
            vmax = hi * 1.10
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(
            Z.T,
            origin='lower',
            cmap=cmap,
            extent=[beta_min, beta_max, v_min, v_max],
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"Shell {i+1}", fontsize=11, weight = 'bold')
        ax.set_xlabel(r"$\beta$ (deg)", fontsize=10)
        if i % ncols == 0:
            ax.set_ylabel(y_label, fontsize=10)

        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=9, length=5, width=1)
        ax.tick_params(axis='both', which='minor', labelsize=8, length=3, width=0.8)

        # (1) Always mark the minimum of the chi2 grid for this shell
        if best is None:
            try:
                jj, kk = np.unravel_index(np.nanargmin(chi2), chi2.shape)
                beta_min_pt = float(beta_array[jj])
                v_min_pt    = float(v_array[kk])
                ax.plot(beta_min_pt, v_min_pt, marker='o', ms=6,
                        mfc='none', mec='lime', mew=1.6, zorder=6)
            except Exception:
                pass

        # (2) If a "best" dict is provided, plot the best + 1σ rectangle
        if best is not None:

            # --- beta center + beta uncertainty ---
            #   - global-beta summaries: beta_star, beta_err_scalar
            #   - your older style: beta / beta_err
            if "beta_star" in best:
                bx_i = float(best.get("beta_star", np.nan))
                be_i = float(best.get("beta_err_scalar", np.nan))
            else:
                bx = best.get("beta", np.nan)
                bx = np.asarray(bx, float)
                bx_i = float(bx) if bx.ndim == 0 else float(bx[i])

                be = best.get("beta_err", np.nan)
                be = np.asarray(be, float)
                be_i = float(be) if be.ndim == 0 else float(be[i])

            # --- v center + v uncertainty (per shell) ---
            vy = np.asarray(best.get("v", np.nan), float)
            by_i = float(vy) if vy.ndim == 0 else float(vy[i])

            ve = np.asarray(best.get("v_err", np.nan), float)
            ve_i = float(ve) if ve.ndim == 0 else float(ve[i])

            # --- draw marker ---
            if np.isfinite(bx_i) and np.isfinite(by_i):
                ax.plot(bx_i, by_i, marker='x', ms=6, mec='cyan',  zorder=7)
                # --- draw 1σ rectangle ---
                if np.isfinite(be_i) and np.isfinite(ve_i) and (be_i > 0) and (ve_i > 0):
                    rect = Rectangle(
                        (bx_i - be_i, by_i - ve_i),
                        2.0 * be_i, 2.0 * ve_i,
                        fill=False, ec='white', lw=1.0, zorder=7
                    )
                    ax.add_patch(rect)        

        # colorbar per panel (matches your earlier version)
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(r'$\log_{10}\,\chi^2$', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        ax.set_autoscale_on(False)
        ax.set_xlim(beta_min, beta_max)
        ax.set_ylim(v_min, v_max)

    # hide unused axes
    for k in range(int(num_shells), nrows * ncols):
        axes_flat[k].axis('off')

    plt.tight_layout()
    #plt.show()
    
    
def _make_conical_ring_masks(shape, center_xy, inc_deg, pa_deg,
                             r_edges_pix, aperture_deg, double_cone, obs_mask):
    """
    Conical *ring* masks for an inclined cone:
    - shells defined in *elliptical* radius (rell) using r_edges_pix,
    - angular cut done in the *intrinsic* azimuth (after deprojection).
    """
    rell, ang_int, _ = _elliptical_radius_and_intrinsic_angle(shape, center_xy, inc_deg, pa_deg)

    masks = []
    half = np.radians(aperture_deg) / 2.0 if (aperture_deg is not None) else np.pi
    # cone centered on intrinsic angle 0 (i.e., along +xr after PA rotation)
    cone_mask = (np.abs(ang_int) <= half)
    if double_cone:
        # add the opposite lobe (±π)
        opp = (np.abs(((ang_int + np.pi) % (2*np.pi)) - np.pi) <= half)
        cone_mask = cone_mask | opp
    cone_mask = cone_mask & obs_mask

    for i in range(len(r_edges_pix) - 1):
        ring = (rell >= r_edges_pix[i]) & (rell < r_edges_pix[i + 1])
        masks.append(ring & cone_mask)

    return masks

# old version, wrong for bicones
# def _make_ring_masks_by_geometry(geometry, shape, center_xy, inc_deg, pa_deg,
#                                  r_edges_pix, aperture_deg, double_cone, obs_mask):
    
#     g = geometry.lower()
#     if g == 'cylindrical':
#         # Disk-like, axisymmetric rings: use q = cos(inc)
#         ny, nx = shape
#         yy, xx = np.indices((ny, nx))
#         xx = xx - center_xy[0]
#         yy = yy - center_xy[1]
#         xr, yr = _rotate_to_pa(xx, yy, pa_deg)

#         q = np.clip(np.cos(np.radians(inc_deg)), 0.01, 1.0)  # DISK: cosine

#         rell = np.sqrt(xr**2 + (yr / q)**2)

#         masks = []
#         base = obs_mask  # no angular cut for disks
#         for i in range(len(r_edges_pix) - 1):
#             ring = (rell >= r_edges_pix[i]) & (rell < r_edges_pix[i + 1])
#             masks.append(ring & base)
#         return masks

#     elif g == 'spherical':
#         return _make_conical_ring_masks(
#           shape, center_xy, inc_deg, pa_deg,
#            r_edges_pix, aperture_deg, double_cone, obs_mask
#        )
#     else:
#         raise ValueError(f"Unsupported geometry '{geometry}' — use 'cylindrical' or 'spherical'.")




# new version good for bi-cones
def _make_ring_masks_by_geometry(geometry, shape, center_xy, inc_deg, pa_deg,
                                r_edges_pix, aperture_deg, double_cone, obs_mask):
    
    geometry = geometry.lower()
    
    if geometry == "cylindrical":
        # unchanged disk logic
        ny, nx = shape
        yy, xx = np.indices((ny, nx))
        xx = xx - center_xy[0]
        yy = yy - center_xy[1]
        # xr, yr = _rotate_to_pa(xx, yy, pa_astro_to_math(pa_deg))
        xr, yr = _rotate_to_pa(xx, yy, pa_deg)
        q = np.clip(np.cos(np.radians(inc_deg)), 0.01, 1.0)
        rell = np.sqrt(xr**2 + (yr / q)**2)
        
        masks = []
        for i in range(len(r_edges_pix) - 1):
            ring = (rell >= r_edges_pix[i]) & (rell < r_edges_pix[i + 1])
            masks.append(ring & obs_mask)
        return masks
    
    else:
        # spherical: handle bicone correctly
        if double_cone:
            beta1, gamma1 = inc_deg, pa_deg
            beta2, gamma2 = 180.0 - inc_deg, (pa_deg + 180.0) % 360.0
            
            return _bicone_masks(
                shape, center_xy,
                beta1, gamma1, beta2, gamma2,
                aperture_deg, r_edges_pix, obs_mask
            )
        else:
            # single cone - use original logic
            return _make_conical_ring_masks(
                shape, center_xy, inc_deg, pa_deg,
                r_edges_pix, aperture_deg, double_cone, obs_mask
            )









def percentile_velocities_from_cube(cube, vel_axis, masks, perc_list):
    n_shells = len(masks)
    out = np.full((n_shells, len(perc_list)), np.nan, dtype=float)
    for i, m in enumerate(masks):
        if not np.any(m): 
            continue
        # prof = np.nan_to_num(np.nansum(cube[:, m], axis=1), nan=0.0)
        prof = np.nansum(cube[:, m], axis=1)
        prof = np.nan_to_num(prof, nan=0.0)
        
        # baseline from the lower part of the distribution (robust)
        baseline = np.nanpercentile(prof, 20)
        prof = prof - baseline
        prof = np.clip(prof, 0.0, None)

        tot = prof.sum()
        if tot <= 0: 
            continue
        cdf = np.cumsum(prof) / tot
        cdf = np.maximum.accumulate(cdf)  # monotonic
        v_at_p = np.interp(perc_list, cdf, vel_axis, left=vel_axis[0], right=vel_axis[-1])
        out[i, :] = v_at_p
    return out


def _nearest_argmin(a):
    return int(np.nanargmin(a))
def _profile_bounds_1d(x, y, delta=1.0):
    
    y = np.asarray(y, float)
    if not np.any(np.isfinite(y)): return (np.nan, np.nan)
    i0 = _nearest_argmin(y)
    ymin = y[i0]; target = ymin + delta

    def side_cross(idx, step):
        i = idx
        while 0 <= i+step < len(y):
            y0, y1 = y[i], y[i+step]
            if np.isfinite(y0) and np.isfinite(y1) and (y0-target)*(y1-target) <= 0:
                x0, x1 = x[i], x[i+step]
                if y1 == y0: return 0.5*(x0+x1)
                t = (target - y0)/(y1 - y0)
                return x0 + t*(x1 - x0)
            i += step
        return np.nan

    return side_cross(i0, -1), side_cross(i0, +1)

def _curvature_sigma_1d(x, y, n_side=2):
    
    if not np.any(np.isfinite(y)): return np.nan
    i0 = _nearest_argmin(y)
    i_lo = max(0, i0 - n_side)
    i_hi = min(len(x), i0 + n_side + 1)
    xs = x[i_lo:i_hi]; ys = y[i_lo:i_hi]
    if len(xs) < 3 or np.any(~np.isfinite(ys)): return np.nan
    A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    a, b, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    if a <= 0: return np.nan
    return np.sqrt(1.0/(2.0*a))

def summarize_independent_shell_fit_with_profiles(chi_squared_map, beta_array, v_array, sigma_scale=1.0):
    
    S = chi_squared_map.shape[0]
    beta_star = np.full(S, np.nan); v_star = np.full(S, np.nan)
    beta_err  = np.full(S, np.nan); v_err  = np.full(S, np.nan)

    for s in range(S):
        chi2 = chi_squared_map[s]
        if not np.any(np.isfinite(chi2)): continue

        ib, jv = np.unravel_index(np.nanargmin(chi2), chi2.shape)
        beta_star[s] = beta_array[ib]
        v_star[s]    = v_array[jv]

        chi_beta = chi2[:, jv]
        chi_v    = chi2[ib, :]

        # --- β profile
        lo, hi = _profile_bounds_1d(beta_array, chi_beta, delta=1.0)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            beta_err[s] = 0.5*(hi - lo)/np.sqrt(2.0)
        else:
            beta_err[s] = _curvature_sigma_1d(beta_array, chi_beta)

        # --- v profile
        lo, hi = _profile_bounds_1d(v_array, chi_v, delta=1.0)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            v_err[s] = 0.5*(hi - lo)/np.sqrt(2.0)
        else:
            v_err[s] = _curvature_sigma_1d(v_array, chi_v)

    print(f"\n=== Independent-shell best fits (Δκ=1 ~{sigma_scale:.0f}σ) ===")

    for s in range(S):
        be = beta_err[s]*sigma_scale
        ve = v_err[s]*sigma_scale
        print(f"Shell {s+1:2d}:  β = {beta_star[s]:5.1f} ± {be:.2f} deg   "
              f"v = {v_star[s]:7.1f} ± {ve:.1f} km/s")
    return dict(beta=beta_star, beta_err=beta_err*sigma_scale, v=v_star, v_err=v_err*sigma_scale)




def summarize_global_beta_with_per_shell_v(chi_squared_map, beta_array, v_array, sigma_scale=1.0):
   
    S, NB, NV = chi_squared_map.shape
    kappa_vs_beta = np.full(NB, np.nan)
    for ib in range(NB):
        mins = np.nanmin(chi_squared_map[:, ib, :], axis=1)  # per-shell min over v
        kappa_vs_beta[ib] = np.nansum(mins)

    ib_best = int(np.nanargmin(kappa_vs_beta))
    beta_star = float(beta_array[ib_best])

    # ---- NEW: 1σ on global β from Δκ=1 on κ_total(β)
    kmin = float(kappa_vs_beta[ib_best])                     # NEW
    dof  = max(S - 1, 1)                                     # NEW: 2S data, (S+1) params
    chi2_red = kmin / dof                                    # NEW
    delta_for_1sigma = chi2_red                              # NEW: inflate Δκ threshold

    lo, hi = _profile_bounds_1d(beta_array, kappa_vs_beta, delta=delta_for_1sigma)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        beta_err_1sigma = 0.5 * (hi - lo) / np.sqrt(2.0)
    else:
        # fallback curvature; also inflate by sqrt(chi2_red) to keep scale consistent
        beta_err_1sigma = _curvature_sigma_1d(beta_array, kappa_vs_beta) * np.sqrt(max(chi2_red, 1.0))  # NEW

    beta_err = beta_err_1sigma * sigma_scale
    # optional: do not claim precision finer than half a β grid step
    if len(beta_array) >= 2:                                  # NEW
        half_step = 0.5 * float(np.min(np.diff(beta_array)))  # NEW
        beta_err = max(beta_err, half_step)                   # NEW

    # per-shell v at fixed β*
    v_star = np.full(S, np.nan)
    v_err  = np.full(S, np.nan)
    for s in range(S):
        prof = chi_squared_map[s, ib_best, :]   # κ_s(v) at fixed β*
        if np.any(np.isfinite(prof)):
            jv = int(np.nanargmin(prof))
            v_star[s] = float(v_array[jv])

            lo, hi = _profile_bounds_1d(v_array, prof, delta=1.0)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                v_err[s] = 0.5 * (hi - lo) / np.sqrt(2.0)
            else:
                v_err[s] = _curvature_sigma_1d(v_array, prof)

    #print("\n=== Global-β best fits ===".format(sigma_scale))
    #print(f"β* = {beta_star:.1f} ± {beta_err:.2f} deg  ")
    for s in range(S):
        print(f"Shell {s+1:2d}: v = {v_star[s]:7.1f} ± {(v_err[s]*sigma_scale):.1f} km/s")

    # Pack a 'best' dict compatible with your plotting utilities
    best = dict(
        beta=np.full(S, beta_star),
        beta_err=np.full(S, beta_err),         
        v=v_star,
        v_err=v_err * sigma_scale,
        beta_star=beta_star,
        beta_err_scalar=beta_err,               
        kappa_vs_beta=kappa_vs_beta,
        ib_best=ib_best
    )
    return best

def global_best_beta_v(chi_squared_map, beta_array, v_array):
    """
    Return the single (β*, v*) that minimizes the *total* κ over shells.
    Ignores NaNs by summing with np.nansum across shells.
    """
    total = np.nansum(chi_squared_map, axis=0)     # (n_beta, n_v)
    ib, jv = np.unravel_index(np.nanargmin(total), total.shape)
    return float(beta_array[ib]), float(v_array[jv])



def summarize_global_beta_mbh(chi_squared_map, beta_array, mbh_array, sigma_scale=1.0):
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)
    ib, jm = np.unravel_index(np.nanargmin(total), total.shape)
    beta_star = float(beta_array[ib]); mbh_star = float(mbh_array[jm])

    # Δκ=1 profiles for simple ~1σ
    lo, hi = _profile_bounds_1d(beta_array, total[:, jm], delta=1.0)
    beta_err = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(beta_array, total[:, jm]))
    lo, hi = _profile_bounds_1d(mbh_array,  total[ib, :],  delta=1.0)
    mbh_err  = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(mbh_array, total[ib, :]))

    beta_err *= sigma_scale; mbh_err *= sigma_scale

    print("\n=== Keplerian rotating-disc best fit (Δκ=1 ~{:.0f}σ) ===".format(sigma_scale))
    print(f"β*  = {beta_star:.2f} ± {beta_err:.2f} deg")
    print(f"M•  = {mbh_star:.3e} ± {mbh_err:.3e} Msun")
    return dict(beta_star=beta_star, beta_err=beta_err, mbh_star=mbh_star, mbh_err=mbh_err, total_kappa=total)


def summarize_global_beta_nsc(chi_squared_map, beta_array, A_array, sigma_scale=1.0):
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)
    ib, jm = np.unravel_index(np.nanargmin(total), total.shape)
    beta_star = float(beta_array[ib]); A_star = float(A_array[jm])

    # Δκ=1 profiles for simple ~1σ
    lo, hi = _profile_bounds_1d(beta_array, total[:, jm], delta=1.0)
    beta_err = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(beta_array, total[:, jm]))
    lo, hi = _profile_bounds_1d(A_array,  total[ib, :],  delta=1.0)
    A_err  = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(A_array, total[ib, :]))

    beta_err *= sigma_scale; A_err *= sigma_scale

    print("\n=== NSC best fit (Δκ=1 ~{:.0f}σ) ===".format(sigma_scale))
    print(f"β*  = {beta_star:.2f} ± {beta_err:.2f} deg")
    print(f"M•  = {A_star:.3e} ± {A_err:.3e} Msun")
    return dict(beta_star=beta_star, beta_err=beta_err, A_star=A_star, A_err=A_err, total_kappa=total)


def summarize_global_beta_plu(chi_squared_map, beta_array, M0_array, sigma_scale=1.0):
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)
    ib, jm = np.unravel_index(np.nanargmin(total), total.shape)
    beta_star = float(beta_array[ib]); M0_star = float(M0_array[jm])

    # Δκ=1 profiles for simple ~1σ
    lo, hi = _profile_bounds_1d(beta_array, total[:, jm], delta=1.0)
    beta_err = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(beta_array, total[:, jm]))
    lo, hi = _profile_bounds_1d(M0_array,  total[ib, :],  delta=1.0)
    M0_err  = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(M0_array, total[ib, :]))

    beta_err *= sigma_scale; M0_err *= sigma_scale

    print("\n=== Plummer best fit (Δκ=1 ~{:.0f}σ) ===".format(sigma_scale))
    print(f"β*  = {beta_star:.2f} ± {beta_err:.2f} deg")
    print(f"M•  = {M0_star:.3e} ± {M0_err:.3e} Msun")
    return dict(beta_star=beta_star, beta_err=beta_err, M0_star=M0_star, M0_err=M0_err, total_kappa=total)


def vkep_astropy(rad_arcsec, theta, phi, pars):
   
    MBH_Msun, scale = pars
    # convert radius to kpc
    r_kpc = np.maximum(np.asarray(rad_arcsec, dtype=float) * scale, 1e-6) 
    # GM/r in SI, then to km/s
    v_ms  = np.sqrt(const.G.value * (MBH_Msun * const.M_sun.value) / (r_kpc * const.kpc.value))
    return v_ms / 1e3



def vnsc_astropy(rad, theta, phi, vnscpars):
   
    A, scale, R_e = vnscpars
    # Convert radius from arcseconds to pc
    r_pc = np.maximum(rad * scale * 1000, 1e-6)  # convert arcsec -> kpc -> pc
    # Calculate enclosed mass using piecewise NSC profile
    # Handle both scalar and array inputs
    if np.isscalar(r_pc):
        if r_pc < R_e:
            # M(<R) = 4π R A for R < R_e
            M_enclosed = 4 * np.pi * r_pc * A
        else:
            # M(<R) = 4π R_e A [1 + log(R/R_e)] for R >= R_e
            M_enclosed = 4 * np.pi * R_e * A * (1 + np.log(r_pc / R_e))
    else:
        # For array inputs
        M_enclosed = np.zeros_like(r_pc)
        # Inner region: R < R_e
        inner_mask = r_pc < R_e
        M_enclosed[inner_mask] = 4 * np.pi * r_pc[inner_mask] * A
        # Outer region: R >= R_e
        outer_mask = r_pc >= R_e
        M_enclosed[outer_mask] = 4 * np.pi * R_e * A * (1 + np.log(r_pc[outer_mask] / R_e))
    
    # Velocity: v = sqrt(GM(<R)/r)
    vel_ms = np.sqrt(const.G.value * M_enclosed * const.M_sun.value / (r_pc * const.pc.value))  # velocity in m/s
    
    return vel_ms / 1e3


def vplummer_astropy(rad, theta, phi, vplummerpars):
   
    
    M0, scale, a = vplummerpars
    
    # Convert radius from arcseconds to pc
    r_pc = rad * scale * 1000  # convert arcsec -> kpc -> pc
    
    # Avoid division by zero at r=0
    r_pc = np.maximum(r_pc, 1e-6)  # minimum radius of 1e-6 pc
    
    # Calculate enclosed mass using Plummer profile: M(<R) = M0 * R^3 / (R^2 + a^2)^(3/2)
    M_enclosed = M0 * (r_pc**3) / ((r_pc**2 + a**2)**(3/2))
    
    
    # Velocity: v = sqrt(GM(<R)/r)
    vel_ms = np.sqrt(const.G.value * (M_enclosed  * const.M_sun.value) / (r_pc * const.pc.value ))  # velocity in m/s

    return vel_ms/1e3





def _plot_v_profile(best_dict, n_shells, title, scale_kpc_per_arcsec, rin_pix, rout_pix, arcsec_per_pix):
    if best_dict is None:
        return

    v_shell = np.asarray(best_dict.get("v", []), float)
    if v_shell.size == 0:
        return

    v_err = best_dict.get("v_err", None)
    if v_err is None:
        v_err = best_dict.get("v_unc", None)
    if v_err is None:
        v_err = np.full_like(v_shell, np.nan, dtype=float)
    else:
        v_err = np.asarray(v_err, float)

    edges_pix = np.linspace(float(rin_pix), float(rout_pix), int(n_shells) + 1)
    rmid_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
    rmid_arcsec = rmid_pix * float(arcsec_per_pix)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', labelsize=12)

    ax.errorbar(
        rmid_arcsec, v_shell, yerr=v_err, color = 'black', 
        fmt="o-", mfc="none", mec="blue", ecolor = 'black', capsize=4, mew=1., lw=1.
    )

    ax.set_xlabel(r"Radius [arcsec]", fontsize=11)
    ax.set_ylabel(r"Velocity [km s$^{-1}$]", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)

    xmax_arc = float(np.nanmax(rmid_arcsec)) if rmid_arcsec.size else 0.0
    ax.set_xlim(0.0, xmax_arc * 1.02 if xmax_arc > 0 else 1.0)

    def a2k(x):
        return x * float(scale_kpc_per_arcsec)

    def k2a(x):
        return x / float(scale_kpc_per_arcsec)

    secax = ax.secondary_xaxis("top", functions=(a2k, k2a))
    secax.set_xlabel("Radius [kpc]", fontsize=11)
    secax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()




def _plot_enclosed_dynamical_mass(best_dict, n_shells, num_shells_selected, title, scale_kpc_per_arcsec, rin_pix, rout_pix, arcsec_per_pix):
    if best_dict is None:
        return

    v_shell = np.asarray(best_dict.get("v", []), float)
    if v_shell.size == 0:
        return

    v_err = np.asarray(best_dict.get("v_err", np.full_like(v_shell, np.nan)), float)

    n_shells = int(n_shells)
    edges_pix = np.linspace(float(rin_pix), float(rout_pix), int(n_shells) + 1)
    edges_arcsec = edges_pix * float(arcsec_per_pix)
    r_out_arcsec = edges_arcsec[1:]

    n = int(min(len(r_out_arcsec), len(v_shell), len(v_err)))
    if n < 1:
        return

    r_out_arcsec = r_out_arcsec[:n]
    v_shell = v_shell[:n]
    v_err = v_err[:n]

    n_sel = int(num_shells_selected)
    if n_sel < 1:
        logger.warning("num_shells_selected must be >= 1 for enclosed mass plot; skipping.")
        return
    if n_sel > n:
        logger.warning(
            "num_shells_selected=%d exceeds available shells=%d; using last shell.",
            int(num_shells_selected), int(n),
        )
        n_sel = n

    r_outer_arcsec = edges_arcsec[n_sel]

    r_kpc = r_out_arcsec * float(scale_kpc_per_arcsec)
    r_m = (r_kpc * u.kpc).to(u.m).value
    v_ms = v_shell * 1e3

    mdyn_msun = np.full_like(v_shell, np.nan, dtype=float)
    mdyn_err_msun = np.full_like(v_shell, np.nan, dtype=float)

    good = np.isfinite(r_m) & (r_m > 0) & np.isfinite(v_ms) & (v_ms > 0)
    if np.any(good):
        mdyn_msun[good] = (v_ms[good] ** 2 * r_m[good] / const.G.value) / const.M_sun.value
        good_err = good & np.isfinite(v_err) & (v_err > 0)
        if np.any(good_err):
            mdyn_err_msun[good_err] = np.abs(mdyn_msun[good_err]) * 2.0 * np.abs(v_err[good_err]) / np.abs(v_shell[good_err])

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', labelsize=12)

    x = r_out_arcsec[:n_sel]
    y = mdyn_msun[:n_sel]
    yerr = np.where(np.isfinite(mdyn_err_msun[:n_sel]), mdyn_err_msun[:n_sel], 0.0)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[ok]
    y = y[ok]
    yerr = yerr[ok]

    logy = np.log10(y)
    logyerr = np.where(
        (yerr > 0) & np.isfinite(yerr),
        yerr / (y * np.log(10.0)),
        0.0
    )

    ax.errorbar(
        x, logy, yerr=logyerr,
        color='black', fmt="o-", mfc="none", mec="blue", ecolor='black',
        capsize=4, mew=1., lw=1.
    )

    ax.set_xlabel(r"Radius [arcsec]", fontsize=11)
    ax.set_ylabel(r"log$_{10}$(Enclosed $M_{dyn}$ [$M_\odot$])", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)

    xmax_arc = float(r_out_arcsec[n_sel - 1]) if np.isfinite(r_out_arcsec[n_sel - 1]) else 0.0
    ax.set_xlim(0.0, xmax_arc * 1.10 if xmax_arc > 0 else 1.0)

    if logy.size:
        ymax = np.nanmax(logy + np.where(np.isfinite(logyerr), logyerr, 0.0))
        if np.isfinite(ymax):
            ymin = np.nanmin(logy - np.where(np.isfinite(logyerr), logyerr, 0.0))
            if np.isfinite(ymin):
                pad = 0.05 * max(ymax - ymin, 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)

    if np.isfinite(r_outer_arcsec):
        ax.axvline(r_outer_arcsec, color="black", ls="--", lw=0.8, alpha=0.6)

    def a2k(x):
        return x * float(scale_kpc_per_arcsec)

    def k2a(x):
        return x / float(scale_kpc_per_arcsec)

    secax = ax.secondary_xaxis("top", functions=(a2k, k2a))
    secax.set_xlabel("Radius [kpc]", fontsize=11)
    secax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    if x.size:
        logger.info("Enclosed dynamical mass profile computed and plotted.")



def plot_beta_profile(best, n_shells, title, rin_pix, rout_pix, arcsec_per_pix, scale_kpc_per_arcsec):
    b = np.asarray(best.get("beta", []), float)
    be = np.asarray(best.get("beta_err", np.full_like(b, np.nan)), float)

    edges_pix = np.linspace(float(rin_pix), float(rout_pix), int(n_shells) + 1)
    edges_arc = edges_pix * arcsec_per_pix
    rmid_arc  = 0.5 * (edges_arc[:-1] + edges_arc[1:])
    xerr_arc  = 0.5 * (edges_arc[1:] - edges_arc[:-1])

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", length=2, direction = 'in',  labelsize = 11)

    ax.errorbar(rmid_arc, b, xerr=xerr_arc, yerr=be, fmt="o-", mfc="none", color = 'black',  ecolor = 'black', mec="blue", capsize=4)
    ax.set_xlabel("Radius [arcsec]", fontsize = 11)
    ax.set_ylabel(r"Inclination $\beta$ [deg]", fontsize = 11)
    ax.set_title(title, fontsize = 11)
    ax.grid(alpha=0.2)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    tick_arc = ax.get_xticks()
    tick_arc = tick_arc[tick_arc >= 0]
    ax.set_xticks(tick_arc)
    ax_top.set_xticks(tick_arc)
    ax_top.set_xticklabels([f"{t * scale_kpc_per_arcsec:.2f}" for t in tick_arc])
    ax_top.set_xlabel("Radius [kpc]", fontsize = 11)
    ax_top.tick_params(which="both", length=2, direction = 'in',  labelsize = 11)



    plt.tight_layout()
    
    
def plot_total_kappa_landscape(chi_squared_map, beta_array, mbh_array, *, best_kepl=None):
    # total κ over shells
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    Z = np.log10(np.where(np.isfinite(total) & (total>0), total, np.nan))
    vmin = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else 0
    vmax = np.nanpercentile(Z, 98) if np.isfinite(Z).any() else 1
    im = ax.imshow(
        Z.T, origin='lower', cmap='inferno',
        extent=[beta_array.min(), beta_array.max(), mbh_array.min(), mbh_array.max()],
        aspect='auto', vmin=vmin, vmax=vmax
    )
    ax.set_xlabel(r'$\beta$ (deg)')
    ax.set_ylabel(r'$M_\bullet\ (M_\odot)$')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r'$\log_{10}\,\Sigma_s \kappa_s$')

    if best_kepl is not None:
        bx  = best_kepl['beta_star']
        bm  = best_kepl['mbh_star']
        ax.plot(bx, bm, marker='s', ms=7, mfc='none', mec='black', mew=1.2, zorder=5)
        # crude 1σ belt along β using your Δκ inflation rule
        total_beta = np.nanmin(total, axis=1)         # Σ_s min_v κ_s(β)
        kmin = np.nanmin(total_beta)
        dof  = max(chi_squared_map.shape[0]-1, 1)
        delta = (kmin/dof)                            # χ²_red scaling you used
        # draw where total_beta <= kmin+delta
        ok = total_beta <= (kmin + delta)
        if np.any(ok):
            be = [beta_array[ok].min(), beta_array[ok].max()]
            ax.hlines(bm, be[0], be[1], colors='cyan', linestyles='-', lw=2, label='~1σ β-range')
            ax.legend(frameon=True)
    ax.set_title('Global landscape: Σ κ(β, M•)')
    plt.tight_layout()
    

def plot_total_kappa_nsc(chi_squared_map, beta_array, A_array, *, best_nsc=None):
    # total κ over shells
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    Z = np.log10(np.where(np.isfinite(total) & (total>0), total, np.nan))
    vmin = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else 0
    vmax = np.nanpercentile(Z, 98) if np.isfinite(Z).any() else 1
    im = ax.imshow(
        Z.T, origin='lower', cmap='inferno',
        extent=[beta_array.min(), beta_array.max(), A_array.min(), A_array.max()],
        aspect='auto', vmin=vmin, vmax=vmax
    )
    ax.set_xlabel(r'$\beta$ (deg)')
    ax.set_ylabel(r'$A (M_\odot/pc)$')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r'$\log_{10}\,\Sigma_s \kappa_s$')

    if best_nsc is not None:
        bx  = best_nsc['beta_star']
        bm  = best_nsc['A_star']
        ax.plot(bx, bm, marker='s', ms=7, mfc='none', mec='blue', mew=1.2, zorder=5)
        # crude 1σ belt along β using your Δκ inflation rule
        total_beta = np.nanmin(total, axis=1)         # Σ_s min_v κ_s(β)
        kmin = np.nanmin(total_beta)
        dof  = max(chi_squared_map.shape[0]-1, 1)
        delta = (kmin/dof)                            # χ²_red scaling you used
        # draw where total_beta <= kmin+delta
        ok = total_beta <= (kmin + delta)
        if np.any(ok):
            be = [beta_array[ok].min(), beta_array[ok].max()]
            ax.hlines(bm, be[0], be[1], colors='cyan', linestyles='-', lw=2, label='~1σ β-range')
            ax.legend(frameon=True)
    ax.set_title('Global landscape: Σ κ(β, A)')
    plt.tight_layout()
    
def plot_total_kappa_plu(chi_squared_map, beta_array, M0_array, *, best_plu=None):
    # total κ over shells
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    Z = np.log10(np.where(np.isfinite(total) & (total>0), total, np.nan))
    vmin = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else 0
    vmax = np.nanpercentile(Z, 98) if np.isfinite(Z).any() else 1
    im = ax.imshow(
        Z.T, origin='lower', cmap='inferno',
        extent=[beta_array.min(), beta_array.max(), M0_array.min(), M0_array.max()],
        aspect='auto', vmin=vmin, vmax=vmax
    )
    ax.set_xlabel(r'$\beta$ (deg)')
    ax.set_ylabel(r'$M0 (M_\odot)$')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r'$\log_{10}\,\Sigma_s \kappa_s$')

    if best_plu is not None:
        bx  = best_plu['beta_star']
        bm  = best_plu['M0_star']
        ax.plot(bx, bm, marker='s', ms=7, mfc='none', mec='blue', mew=1.2, zorder=5)
        # crude 1σ belt along β using your Δκ inflation rule
        total_beta = np.nanmin(total, axis=1)         # Σ_s min_v κ_s(β)
        kmin = np.nanmin(total_beta)
        dof  = max(chi_squared_map.shape[0]-1, 1)
        delta = (kmin/dof)                            # χ²_red scaling you used
        # draw where total_beta <= kmin+delta
        ok = total_beta <= (kmin + delta)
        if np.any(ok):
            be = [beta_array[ok].min(), beta_array[ok].max()]
            ax.hlines(bm, be[0], be[1], colors='cyan', linestyles='-', lw=2, label='~1σ β-range')
            ax.legend(frameon=True)
    ax.set_title('Global landscape: Σ κ(β, M0)')
    plt.tight_layout()



def show_vfield_comparison(obs, model_best, vel_axis, *, center_xy, pixscale,
                           title=None, res_vlim=None, xlimit = None, ylimit = None):
    obs_vpct=97
    # weight cube before comparison
    model_best.weight_cube(obs.cube['data'])
    model_best.generate_cube(weights=model_best.cube['weights'])
    model_best.kin_maps_cube(fluxthr=1e-50)

    vobs = obs.maps['vel']
    vmod = model_best.maps['vel']
    vres = vobs - vmod

    ny, nx = vobs.shape
    arcsec_per_pix = pixscale
    x_arc = (np.arange(nx) - center_xy[0]) * arcsec_per_pix
    y_arc = (np.arange(ny) - center_xy[1]) * arcsec_per_pix
    extent = [x_arc[0], x_arc[-1], y_arc[0], y_arc[-1]]

    # symmetric stretch for Obs/Model
    vmax = float(max(np.nanpercentile(np.abs(vobs), obs_vpct), 1.0))
    vmin = -vmax

    # residual stretch (smaller on purpose)
    if res_vlim is None:
        vmax_res = float(max(np.nanpercentile(np.abs(vres), 98), 1.0))
    else:
        vmax_res = float(res_vlim)             # e.g., 200 (km/s)
    vmin_res = -vmax_res

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, constrained_layout=True)
    for ax, M, label in zip(axes, [vobs, vmod, vres], ['Obs', 'Model', 'Residual']):
        if label == 'Residual':
            im = ax.imshow(M, origin='lower', extent=extent,
                           vmin=vmin_res, vmax=vmax_res, cmap='RdBu_r', interpolation='nearest')
            if xlimit is not None and ylimit is not None:
                ax.set_xlim(xlimit[0], xlimit[1])
                ax.set_ylim(ylimit[0], ylimit[1])
                
        else:
            im = ax.imshow(M, origin='lower', extent=extent,
                           vmin=vmin, vmax=vmax, cmap='coolwarm', interpolation='nearest')
            if xlimit is not None and ylimit is not None:
                ax.set_xlim(xlimit[0], xlimit[1])
                ax.set_ylim(ylimit[0], ylimit[1])
        ax.scatter([0.0], [0.0], s=80, c='k', marker='*', edgecolor='white', zorder=3)
        ax.set_xlabel(r'$\Delta X$ [arcsec]'); ax.set_ylabel(r'$\Delta Y$ [arcsec]')
        ax.set_title(label)
        cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
        cbar.set_label(r'$v_{\rm los}$ (km s$^{-1}$)' if label != 'Residual'
                       else r'$\Delta v_{\rm los}$ (km s$^{-1}$)')

    if title:
        fig.suptitle(title, y=1.02)




def plot_chi2_vs_beta_global(chi_squared_map, beta_array, v_array, *,
                             USE_GLOBAL_BETA=True,
                             reduce_v="min",          # "min" or "percentile"
                             v_percentile=10.0,       # used only if reduce_v="percentile"
                             combine_shells="sum",    # "sum" or "median"
                             logy=True,
                             title=r"$\chi^2(\beta)$",
                             show=True):
    """
    If USE_GLOBAL_BETA=True -> plots ONE curve: sum_shells(min_v chi2[shell,beta,v]).
    If USE_GLOBAL_BETA=False -> plots per-shell curves + combined curve (debug style).
    """

    chi = np.asarray(chi_squared_map, float)  # (n_shells, n_beta, n_v)
    beta = np.asarray(beta_array, float)
    v = np.asarray(v_array, float)

    if chi.ndim != 3:
        raise ValueError(f"chi_squared_map must be 3D (n_shells,n_beta,n_v). Got {chi.shape}")
    n_shells, n_beta, n_v = chi.shape
    if beta.size != n_beta:
        raise ValueError(f"beta_array length {beta.size} != chi n_beta {n_beta}")
    if v.size != n_v:
        raise ValueError(f"v_array length {v.size} != chi n_v {n_v}")

    # reduce over v -> chi_beta_per_shell: (n_shells, n_beta)
    if reduce_v == "min":
        chi_beta_per_shell = np.nanmin(chi, axis=2)
    elif reduce_v == "percentile":
        chi_beta_per_shell = np.nanpercentile(chi, v_percentile, axis=2)
    else:
        raise ValueError("reduce_v must be 'min' or 'percentile'")

    # combine shells -> one curve chi_beta_comb: (n_beta,)
    if combine_shells == "sum":
        chi_beta_comb = np.nansum(chi_beta_per_shell, axis=0)
    elif combine_shells == "median":
        chi_beta_comb = np.nanmedian(chi_beta_per_shell, axis=0)
    else:
        raise ValueError("combine_shells must be 'sum' or 'median'")
    # best beta from the global curve
    jbest = int(np.nanargmin(chi_beta_comb))
    beta_best = float(beta[jbest])
    # --- 1σ region based on Δχ² threshold ---
    kmin = float(chi_beta_comb[jbest])
    dof  = max(n_shells - 1, 1)
    delta = kmin / dof
    
    ok = chi_beta_comb <= (kmin + delta)
    if np.any(ok):
        beta_lo = beta[ok].min()
        beta_hi = beta[ok].max()

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)
    ax.tick_params(axis='both', labelsize=11)
    ax.minorticks_on()


    # only show per-shell curves if NOT global-beta mode
    if not USE_GLOBAL_BETA:
        for s in range(n_shells):
            ax.plot(beta, np.log10(chi_beta_per_shell[s]), alpha=0.25, lw=1.0, c = 'black')

    ax.plot(beta, np.log10(chi_beta_comb), lw=0.7, c = 'black')
    ax.scatter(beta, np.log10(chi_beta_comb), marker = 'o', s = 30, facecolors = 'none', color = 'blue')

    ax.axvline(beta_best, ls="--", lw=1.5, color = 'red')
    ax.scatter([beta_best], np.log10([chi_beta_comb[jbest]]),  s=40, zorder=5, color = 'red')
    
    if np.any(ok):
        ax.axvspan(beta_lo, beta_hi, color="cyan", alpha=0.2, zorder=0)


    ax.set_xlabel(r"Inclination $\beta$ (deg)", fontsize = 11)
    ax.set_ylabel(r"Log($\chi^2$)", fontsize = 11)
    ax.set_title(title, fontsize = 11)



    # if logy:
    #     good = np.isfinite(chi_beta_comb) & (chi_beta_comb > 0)
    #     if np.any(good):
    #         ax.set_yscale("log")

    plt.tight_layout()
    


    return beta_best, beta, chi_beta_comb, chi_beta_per_shell


def plot_corner_kappa(chi_squared_map, beta_array, mbh_array, *, best_kepl=None):
    """
    Corner-style view of the Keplerian fit:
      - Center: log10 total κ(β, M•)
      - Top:    κ_total(β) = min_M• κ(β, M•)
      - Right:  κ_total(M•) = min_β κ(β, M•)
    One-sigma bands are drawn using Δκ = χ²_red at the profile minimum, with
    χ²_red computed using dof = (n_shells - 1) as in your summary util.
    """
    # aggregate over shells
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)
    # 1D profiles
    prof_beta = np.nanmin(total, axis=1)  # min over M• at each β
    prof_mbh  = np.nanmin(total, axis=0)  # min over β at each M•
    S = chi_squared_map.shape[0]
    dof = max(S - 1, 1)

    # Δκ thresholds for ~1σ bands
    kmin_b = float(np.nanmin(prof_beta))
    kmin_m = float(np.nanmin(prof_mbh))
    delta_b = kmin_b / dof
    delta_m = kmin_m / dof

    # handy locators for 1σ spans
    ok_b = prof_beta <= (kmin_b + delta_b)
    ok_m = prof_mbh  <= (kmin_m + delta_m)
    
    logM = np.log10(mbh_array)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7.8, 6.6), dpi=150)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1.4], height_ratios=[1.4, 4],
                           hspace=0.06, wspace=0.2)

    # Top: κ_total(β)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.scatter(beta_array, np.log10(prof_beta), s = 8, marker = 'o', color = 'white', edgecolor = 'black')
    ax_top.plot(beta_array, np.log10(prof_beta),  color = 'black', lw = 1)

    ax_top.set_xlim(beta_array.min(), beta_array.max())
    ax_top.set_xticklabels([])
    ax_top.set_ylabel(r'$\log_{10}\,\min_{M_\bullet}\Sigma\kappa$')
    # 1σ span in β
    if np.any(ok_b):
        be0, be1 = beta_array[ok_b].min(), beta_array[ok_b].max()
        ax_top.axvspan(be0, be1, color='cyan', alpha=0.25, lw=1)

    # Right: κ_total(M•)
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.scatter(np.log10(mbh_array), np.log10(prof_mbh),  s = 8, marker = 'o', color = 'white', edgecolor = 'black')
    ax_right.plot(np.log10(mbh_array), np.log10(prof_mbh), color = 'black',lw = 1)

    ax_right.set_ylim(ax_right.get_ylim()[0], ax_right.get_ylim()[1])
    ax_right.set_yticklabels([])
    ax_right.set_xlabel(r'$\log_{10}\,M_\bullet\ (M_\odot)$')
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.set_ylabel(r'$\log_{10}\,\min_{\beta}\Sigma\kappa$', rotation=270, labelpad=12)

    # 1σ span in M•
    if np.any(ok_m):
        me0, me1 = np.log10([mbh_array[ok_m].min(), mbh_array[ok_m].max()])
        ax_right.axvspan(me0, me1, color='cyan', alpha=0.25, lw=1)

    # Center: 2D landscape
    ax = fig.add_subplot(gs[1, 0])
    Z = np.log10(np.where(np.isfinite(total) & (total > 0), total, np.nan))
    # Z -=  np.nanmin(Z)
    vmin = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else 0
    vmax = np.nanpercentile(Z, 98) if np.isfinite(Z).any() else 1
    im = ax.imshow(
                    Z.T, origin='lower', cmap='inferno',
                    extent=[beta_array.min(), beta_array.max(), logM.min(), logM.max()],
                    aspect='auto', vmin=vmin, vmax=vmax
                )
    ax.set_xlabel(r'$\beta$ (deg)')
    ax.set_ylabel(r'$\log_{10}\,M_\bullet\ (M_\odot)$')
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    cbar.set_label(r'$\log_{10}\,\Sigma_s \kappa_s$')

    # Best point markers + guide lines
    if best_kepl is not None:
        bx, bm = best_kepl['beta_star'], best_kepl['mbh_star']
        ax.plot(bx, np.log10(bm), marker='s', ms=7, mfc='none', mec='blue', mew=1.2)
        # guide lines into the marginals
        ax.axvline(bx, ls='--', lw=1.1, c='white', alpha=0.9)
        ax.axhline(np.log10(bm), ls='--', lw=1.1, c='white', alpha=0.9)

        ax_top.axvline(bx, ls='--', lw=1.1, c='red', alpha=0.9)
        ax_right.axvline(np.log10(bm), ls='--', lw=1.1, c='red', alpha=0.9)

    ax_top.grid(alpha=0.25); ax_right.grid(alpha=0.25); ax.grid(alpha=0.15)
    # fig.suptitle('Corner view', y=0.98, fontsize=12)
    #plt.show()


def plot_corner_kappa_nsc(chi_squared_map, beta_array, A_array, *, best_nsc=None):
    # aggregate over shells
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)
    # 1D profiles
    prof_beta = np.nanmin(total, axis=1)  # min over M• at each β
    prof_A  = np.nanmin(total, axis=0)  # min over β at each M•
    S = chi_squared_map.shape[0]
    dof = max(S - 1, 1)

    # Δκ thresholds for ~1σ bands
    kmin_b = float(np.nanmin(prof_beta))
    kmin_m = float(np.nanmin(prof_A))
    delta_b = kmin_b / dof
    delta_m = kmin_m / dof

    # handy locators for 1σ spans
    ok_b = prof_beta <= (kmin_b + delta_b)
    ok_m = prof_A  <= (kmin_m + delta_m)
    
    logA = np.log10(A_array)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7.8, 6.6), dpi=150)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1.4], height_ratios=[1.4, 4],
                           hspace=0.06, wspace=0.2)

    # Top: κ_total(β)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.scatter(beta_array, np.log10(prof_beta), s = 8, marker = 'o', color = 'white', edgecolor = 'black')
    ax_top.plot(beta_array, np.log10(prof_beta),  color = 'black', lw = 1)

    ax_top.set_xlim(beta_array.min(), beta_array.max())
    ax_top.set_xticklabels([])
    ax_top.set_ylabel(r'$\log_{10}\,\min_{M_\bullet}\Sigma\kappa$')
    # 1σ span in β
    if np.any(ok_b):
        be0, be1 = beta_array[ok_b].min(), beta_array[ok_b].max()
        ax_top.axvspan(be0, be1, color='cyan', alpha=0.25, lw=1)

    # Right: κ_total(M•)
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.scatter(np.log10(A_array), np.log10(prof_A),  s = 8, marker = 'o', color = 'white', edgecolor = 'black')
    ax_right.plot(np.log10(A_array), np.log10(prof_A), color = 'black',lw = 1)

    ax_right.set_ylim(ax_right.get_ylim()[0], ax_right.get_ylim()[1])
    ax_right.set_yticklabels([])
    ax_right.set_xlabel(r'$\log_{10}\,M0 (M_\odot)$')  # (or A for NSC)

    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    
    # Put label on the RIGHT, centered, and push it OUTSIDE the axes so it never overlaps.
    ax_right.set_ylabel(r'$\log_{10}\,\min_{\beta}\Sigma\kappa$', rotation=270, va="center")
    ax_right.yaxis.set_label_coords(1.25, 0.5)  # <-- key line: move label further right

    # 1σ span in M•
    if np.any(ok_m):
        me0, me1 = np.log10([A_array[ok_m].min(), A_array[ok_m].max()])
        ax_right.axvspan(me0, me1, color='cyan', alpha=0.25, lw=1)

    # Center: 2D landscape
    ax = fig.add_subplot(gs[1, 0])
    Z = np.log10(np.where(np.isfinite(total) & (total > 0), total, np.nan))
    # Z -=  np.nanmin(Z)
    vmin = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else 0
    vmax = np.nanpercentile(Z, 98) if np.isfinite(Z).any() else 1
    im = ax.imshow(
                    Z.T, origin='lower', cmap='inferno',
                    extent=[beta_array.min(), beta_array.max(), logA.min(), logA.max()],
                    aspect='auto', vmin=vmin, vmax=vmax
                )
    ax.set_xlabel(r'$\beta$ (deg)')
    ax.set_ylabel(r'$\log_{10}\,A (M_\odot)/pc$')
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    # cbar.set_label(r'$\log_{10}\,\Sigma_s \kappa_s$')

    # Best point markers + guide lines
    if best_nsc is not None:
        bx, bm = best_nsc['beta_star'], best_nsc['A_star']
        ax.plot(bx, np.log10(bm), marker='s', ms=7, mfc='none', mec='blue', mew=1.2)
        # guide lines into the marginals
        ax.axvline(bx, ls='--', lw=1.1, c='white', alpha=0.9)
        ax.axhline(np.log10(bm), ls='--', lw=1.1, c='white', alpha=0.9)

        ax_top.axvline(bx, ls='--', lw=1.1, c='red', alpha=0.9)
        ax_right.axvline(np.log10(bm), ls='--', lw=1.1, c='red', alpha=0.9)

    ax_top.grid(alpha=0.25); ax_right.grid(alpha=0.25); ax.grid(alpha=0.15)
    fig.subplots_adjust(right=0.9)
    #plt.show()

def percentile_scatter_per_shell_best(
    best, obs, vel_axis, center_xy, pa_deg,
    n_shells, r_min_pix, r_max_pix,
    aperture_deg, double_cone,
    *, pixscale, nrebin, scale,
    min_pixels_per_shell=2,
    perc=(0.01, 0.99),
    ncloud=100_000
):
 
    # --- context ---
    geometry = _ctx_get("geometry", required=True)
    FIT_MODE = _ctx_get("FIT_MODE", required=True)
    gamma_model = _ctx_get("gamma_model", required=True)

    # KPC_PER_ARCSEC: prefer context, else use passed `scale`
    KPC_PER_ARCSEC = _ctx_get("KPC_PER_ARCSEC", default=None)
    if KPC_PER_ARCSEC is None:
        # your scripts often call this `scale`; it's kpc/arcsec
        KPC_PER_ARCSEC = float(scale)

    arcsec_per_pix = float(pixscale) * float(nrebin)

    radii_arcsec = []
    radii_kpc    = []
    all_obs = []
    all_mdl = []

    perc = tuple(np.asarray(perc, float))

    for s in range(int(n_shells)):
        b = float(np.asarray(best["beta"])[s])
        v = float(np.asarray(best["v"])[s])

        if not (np.isfinite(b) and np.isfinite(v)):
            radii_arcsec.append(np.nan); radii_kpc.append(np.nan)
            all_obs.append(np.full(len(perc), np.nan))
            all_mdl.append(np.full(len(perc), np.nan))
            continue

        # build model for this shell
        # (keep your original behavior for arctan if rt_best exists in best dict)
        if (geometry.lower() == "cylindrical") and (FIT_MODE == "disk_arctan"):
            # try to use rt from context, fall back to RT_ARCSEC if present
            rt_best = best.get("rt", None)
            if rt_best is None:
                rt_best = _ctx_get("RT_ARCSEC", required=True)
            model_s = make_mod(
                b, v, obs, gamma_model,
                vel3_pars_override=[float(v), float(rt_best)],
                ncloud=int(ncloud)
            )
        else:
            disc_cube = _ctx_get("disc_cube", default=None)
            model_s1 = make_mod(b, v, obs, gamma_model, ncloud=int(ncloud))
            if double_cone:
                beta_flip = 180.0 - b
                gamma_flip = (gamma_model + 180.0) % 360.0
                
                model_s1 = make_mod(b,        v, obs, gamma_model, ncloud=int(ncloud))
                model_s2 = make_mod(beta_flip, v, obs, gamma_flip,  ncloud=int(ncloud))
                
                # temporarily override gamma for second lobe
                # _FIT_CTX["gamma_model"] = gamma_flip
                # model_s2 = make_mod(beta_flip, v, obs, gamma_flip, ncloud=int(ncloud))
                # _FIT_CTX["gamma_model"] = gamma_model  # restore
            
                
                # combine all components
                if disc_cube is not None:
                    combined_cube = disc_cube + model_s1.cube["data"] + model_s2.cube["data"]
                else:
                    combined_cube = model_s1.cube["data"] + model_s2.cube["data"]
                
                # create a temporary model object with combined cube
                model_s = make_observed_like(obs, combined_cube)
                model_s.cube = {"data": combined_cube}
            else:
                if disc_cube is not None:
                    combined_cube = disc_cube + model_s1.cube["data"]
                    model_s = make_observed_like(obs, combined_cube)  
                    model_s.cube = {"data": combined_cube}
                else:
                    model_s = model_s1

        shape = obs.cube["data"].shape[1:]
        model_mask = _finite_mask_from_cube(model_s.cube["data"])
        base_mask  = model_mask

        ap_for_edges = None if geometry.lower() == "cylindrical" else aperture_deg

        # ---- RESTORE LEGACY PA CONVENTION ----
        pa_math = pa_astro_to_mask_angle(pa_deg, geometry)

        r_edges_pix = _shell_edges_from_mask(
            shape, center_xy, b, pa_math, base_mask,
            r_min_pix, r_max_pix, n_shells,
            aperture_deg=ap_for_edges, double_cone=double_cone
        )
        ring_masks = _make_ring_masks_by_geometry(
            geometry, shape, center_xy, b, pa_math,
            r_edges_pix, aperture_deg, double_cone, base_mask
        )

        npx = int(np.count_nonzero(ring_masks[s]))
        if npx < int(min_pixels_per_shell):
            radii_arcsec.append(np.nan); radii_kpc.append(np.nan)
            all_obs.append(np.full(len(perc), np.nan))
            all_mdl.append(np.full(len(perc), np.nan))
            continue

        rmid_pix = 0.5 * (r_edges_pix[s] + r_edges_pix[s + 1])
        rmid_arc = rmid_pix * arcsec_per_pix
        rmid_kpc = rmid_arc * float(KPC_PER_ARCSEC)
        radii_arcsec.append(rmid_arc); radii_kpc.append(rmid_kpc)

        p_obs = percentile_velocities_from_cube(
            obs.cube["data"], np.asarray(vel_axis, float), [ring_masks[s]], perc
        )[0]
        p_mdl = percentile_velocities_from_cube(
            model_s.cube["data"], np.asarray(vel_axis, float), [ring_masks[s]], perc
        )[0]
        all_obs.append(p_obs); all_mdl.append(p_mdl)

    radii_arcsec = np.asarray(radii_arcsec, float)
    radii_kpc    = np.asarray(radii_kpc,    float)
    all_obs = np.asarray(all_obs, float)
    all_mdl = np.asarray(all_mdl, float)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=250)
    ax.tick_params(axis='both', labelsize=12)


    for k, p in enumerate(perc):
        sc_obs = ax.scatter(
            radii_arcsec, all_obs[:, k],
            s=36, marker="o", linewidths=0.0,
            label=f"Obs p{int(round(100*p))}", zorder=3
        )

        # match the model marker edge color to obs marker face color
        fc = sc_obs.get_facecolors()
        if fc is not None and fc.size > 0:
            this_color = fc[0]
        else:
            cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
            this_color = cycle[k % len(cycle)]

        ax.scatter(
            radii_arcsec, all_mdl[:, k],
            s=60, marker="o", facecolors="none",
            linewidths=1.4, edgecolors=this_color,
            label=f"Model p{int(round(100*p))}", zorder=2
        )

    ax.set_xlabel("Radius (arcsec)", fontsize=12)
    ax.set_ylabel("Percentile velocity (km s$^{-1}$)", fontsize=12)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.legend(ncol=2, fontsize=9, frameon=True, framealpha=0.9)

    # secondary x axis: kpc
    def a2k(x): return x * float(KPC_PER_ARCSEC)
    def k2a(x): return x / float(KPC_PER_ARCSEC)

    secx = ax.secondary_xaxis("top", functions=(a2k, k2a))
    secx.set_xlabel("Radius (kpc)", fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    plt.tight_layout()
    #plt.show()

    return dict(
        r_arcsec=radii_arcsec, r_kpc=radii_kpc,
        obs_percentiles=all_obs, mdl_percentiles=all_mdl,
        perc=np.array(perc, float),
    )

def kepler_rc_vs_percentiles(
    best_kepl,
    *,
    obs,
    vel_axis,
    center_xy,
    pa_deg,
    n_shells,
    r_min_pix,
    r_max_pix,
    aperture_deg,
    double_cone,
    pixscale,
    nrebin,
    scale,
    perc=None,
):
    

    # --- get geometry from km context ---
    geom = _ctx_get("geometry", required=True).lower()

    beta_best = float(best_kepl["beta_star"])
    MBH = float(best_kepl["mbh_star"])

    # Build best Kepler model (your make_mod handles disk_kepler when context FIT_MODE says so,
    # but here we call it explicitly with MBH)
    model_best = make_mod(beta_best, MBH, obs, _ctx_get("gamma_model", required=True))

    # Build model-defined shells at beta_best
    shape = obs.cube["data"].shape[1:]
    base_mask = _finite_mask_from_cube(model_best.cube["data"])

    ap_for_edges = None if geom == "cylindrical" else aperture_deg

    # Legacy PA convention conversion handled inside pa_astro_to_mask_angle
    pa_math = pa_astro_to_mask_angle(pa_deg, geom)

    r_edges_pix = _shell_edges_from_mask(
        shape, center_xy, beta_best, pa_math, base_mask,
        r_min_pix, r_max_pix, n_shells,
        aperture_deg=ap_for_edges, double_cone=double_cone
    )

    ring_masks = _make_ring_masks_by_geometry(
        geom, shape, center_xy, beta_best, pa_math,
        r_edges_pix, aperture_deg, double_cone, base_mask
    )

    # Percentiles on these shells
    if perc is None:
        perc = _ctx_get("perc", default=(0.01, 0.99))
    perc = np.asarray(perc, float)

    p_obs = percentile_velocities_from_cube(obs.cube["data"], vel_axis, ring_masks, perc)

    # Radii in arcsec
    arcsec_per_pix = float(pixscale) * float(nrebin)
    rmid_pix = 0.5 * (r_edges_pix[:-1] + r_edges_pix[1:])
    r_arcsec = rmid_pix * arcsec_per_pix

    # Kepler curve (intrinsic circular speed)
    v_kep = vkep_astropy(r_arcsec, None, None, pars=[MBH, float(scale)])

    # Simple LOS expectation band
    v_los_max = v_kep * np.sin(np.radians(beta_best))

    # Envelope from provided percentiles
    lo = np.nanmin(p_obs, axis=1)
    hi = np.nanmax(p_obs, axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)

    ax.fill_between(
        r_arcsec, lo, hi,
        alpha=0.25,
        label=f"Obs p[{int(100*np.min(perc))}–{int(100*np.max(perc))}] (LOS)",
        step="mid"
    )
    ax.plot(r_arcsec, +v_los_max, lw=2, label=r"$\pm v_{\rm kep}(r)\sin\beta$")
    ax.plot(r_arcsec, -v_los_max, lw=2)

    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel(r"Velocity (km s$^{-1}$)")
    ax.set_title(fr"Kepler summary: β={beta_best:.1f}°,  $M_\bullet={MBH:.2e}\ M_\odot$")
    ax.grid(alpha=0.3)

    # top axis in kpc
    KPC_PER_ARCSEC = float(scale)

    def a2k(x): return x * KPC_PER_ARCSEC
    def k2a(x): return x / KPC_PER_ARCSEC

    secx = ax.secondary_xaxis("top", functions=(a2k, k2a))
    secx.set_xlabel("Radius (kpc)")

    ax.legend()
    plt.tight_layout()
    #plt.show()
    
def show_shells_overlay(
    cube_obs, center_xy, inc_deg, pa_deg,
    n_shells, r_min_pix, r_max_pix,
    aperture_deg, double_cone,
    *, pixscale, nrebin, scale, title=None,
    cmap='inferno',
    cube_model=None, mask_mode="obs", edges_mode="obs",
    debug_intrinsic=False, xlimit=None, ylimit=None
):
   
    # --- get geometry from km context ---
    geom = _ctx_get("geometry", required=True).lower()

    # ---- get math angle for MAJOR axis (restores your legacy convention) ----
    pa_math = pa_astro_to_mask_angle(pa_deg, geom)

    # aperture only applies for cones
    ap_for_edges = None if geom == "cylindrical" else aperture_deg

    # -----------------------------
    # Masks
    # -----------------------------
    shape = cube_obs.shape[1:]  # (ny, nx)

    obs_mask = _finite_mask_from_cube(cube_obs)
    if cube_model is not None:
        model_mask = _finite_mask_from_cube(cube_model)
    else:
        model_mask = np.zeros_like(obs_mask, dtype=bool)

    def _pick(mode, obs_m, mdl_m):
        if mode == "model":
            return mdl_m
        if mode == "intersection":
            return obs_m & mdl_m
        if mode == "union":
            return obs_m | mdl_m
        return obs_m  # default "obs"

    base_mask = _pick(mask_mode, obs_mask, model_mask)
    edge_mask = _pick(edges_mode, obs_mask, model_mask)

    # -----------------------------
    # Shell edges
    # -----------------------------
    r_edges_pix = _shell_edges_from_mask(
        shape, center_xy, inc_deg, pa_math, edge_mask,
        r_min_pix, r_max_pix, n_shells,
        aperture_deg=ap_for_edges, double_cone=double_cone
    )

    # -----------------------------
    # Flux map + axes
    # -----------------------------
    flux_map = np.nansum(cube_obs, axis=0)
    ny, nx = shape
    arcsec_per_pix = float(pixscale) * float(nrebin)

    x_arc = (np.arange(nx) - center_xy[0]) * arcsec_per_pix
    y_arc = (np.arange(ny) - center_xy[1]) * arcsec_per_pix
    extent_arcsec = [x_arc[0], x_arc[-1], y_arc[0], y_arc[-1]]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)

    # pick a safe lower bound for log10
    flux = np.array(flux_map, dtype=float)

    valid = np.isfinite(flux) & (flux > 0)
    
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("No positive finite flux values for display.")
    
    logflux = np.full_like(flux, np.nan)
    logflux[valid] = np.log10(flux[valid])
    
    # percentiles computed ONLY on valid log pixels
    vmin, vmax = np.nanpercentile(logflux[valid], [1, 99.5])
    
    im = ax.imshow(
        logflux,
        origin="lower",
        cmap=cmap,
        extent=extent_arcsec,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
    cbar.set_label("Log(Flux)", fontsize=11)
    cbar.ax.tick_params(labelsize=11)   

    xmin, xmax, ymin, ymax = im.get_extent()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    clip_rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          transform=ax.transData, visible=False)
    ax.add_patch(clip_rect)

    # -----------------------------
    # Projection factor q
    # -----------------------------
    if geom == "cylindrical":
        q = np.clip(np.cos(np.radians(inc_deg)), 0.01, 1.0)
    else:
        q = np.clip(np.sin(np.radians(inc_deg)), 0.01, 1.0)

    r_edges_arcsec = r_edges_pix * arcsec_per_pix

    # helper rotation back to sky plot frame
    def rotback(xr, yr, ang_deg):
        th = np.radians(ang_deg)
        X = xr*np.cos(th) - yr*np.sin(th)
        Y = xr*np.sin(th) + yr*np.cos(th)
        return X, Y

    # -----------------------------
    # Draw shells
    # -----------------------------
    if geom == "cylindrical":
        for r_arc in r_edges_arcsec[1:]:
            a = float(r_arc)
            b = float(q * r_arc)
            e = Ellipse((0.0, 0.0), width=2*a, height=2*b, angle=float(pa_math),
                        fill=False, edgecolor="black", linewidth=1.2, zorder=6)
            e.set_clip_path(clip_rect)
            ax.add_patch(e)

    else:
        half = 0.5 * np.radians(aperture_deg) if aperture_deg is not None else np.pi
        Rmax = float(r_edges_arcsec[-1])

        # boundary rays
        if aperture_deg is not None:
            for sgn in (-1, +1):
                xr = Rmax * np.cos(sgn * half)
                yr = Rmax * np.sin(sgn * half)
                X2, Y2 = rotback(xr, q*yr, pa_math)
                ax.plot([0.0, X2], [0.0, Y2], color="black", lw=1.2, zorder=6)

        # circular arcs (in intrinsic coords)
        nphi = 400
        phi = np.linspace(-half, +half, nphi)
        for r_arc in r_edges_arcsec[1:]:
            xr = r_arc * np.cos(phi)
            yr = r_arc * np.sin(phi)
            Xc, Yc = rotback(xr, q*yr, pa_math)
            ax.plot(Xc, Yc, color="white", lw=1.2, zorder=6)

            if double_cone:
                xr2 = r_arc * np.cos(phi + np.pi)
                yr2 = r_arc * np.sin(phi + np.pi)
                Xc2, Yc2 = rotback(xr2, q*yr2, pa_math)
                ax.plot(Xc2, Yc2, color="white", lw=1.2, zorder=6)

    # -----------------------------
    # Labels / limits
    # -----------------------------
    ax.scatter([0.0], [0.0], s=110, c="cyan", marker="*", edgecolor="black", zorder=7)
    ax.set_xlabel(r"$\Delta$ X [arcsec]", fontsize=11)
    ax.set_ylabel(r"$\Delta$ Y [arcsec]", fontsize=11)
    ax.set_aspect("equal")

    if title is None:
        kind = "disk" if geom == "cylindrical" else "conical"
        title = f"{kind} shells (β={inc_deg:.1f}°, PA_in={float(pa_deg)%360:.1f}°)"
    ax.set_title(title, fontsize = 11)

    if (xlimit is not None) and (ylimit is not None):
        ax.set_xlim(xlimit[0], xlimit[1])
        ax.set_ylim(ylimit[0], ylimit[1])

    ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1)
    ax.tick_params(axis="both", which="minor", labelsize=8, length=3, width=0.8)

    plt.tight_layout()
    
def plot_corner_kappa_plu(chi_squared_map, beta_array, M0_array, *, best_plu=None):
    # aggregate over shells
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_mbh)
    # 1D profiles
    prof_beta = np.nanmin(total, axis=1)  # min over M• at each β
    prof_M0  = np.nanmin(total, axis=0)  # min over β at each M•
    S = chi_squared_map.shape[0]
    dof = max(S - 1, 1)

    # Δκ thresholds for ~1σ bands
    kmin_b = float(np.nanmin(prof_beta))
    kmin_m = float(np.nanmin(prof_M0))
    delta_b = kmin_b / dof
    delta_m = kmin_m / dof

    # handy locators for 1σ spans
    ok_b = prof_beta <= (kmin_b + delta_b)
    ok_m = prof_M0  <= (kmin_m + delta_m)
    
    logM0 = np.log10(M0_array)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7.8, 6.6), dpi=150)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1.4], height_ratios=[1.4, 4],
                           hspace=0.06, wspace=0.2)

    # Top: κ_total(β)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.scatter(beta_array, np.log10(prof_beta), s = 8, marker = 'o', color = 'white', edgecolor = 'black')
    ax_top.plot(beta_array, np.log10(prof_beta),  color = 'black', lw = 1)

    ax_top.set_xlim(beta_array.min(), beta_array.max())
    ax_top.set_xticklabels([])
    ax_top.set_ylabel(r'$\log_{10}\,\min_{M0}\Sigma\kappa$')
    # 1σ span in β
    if np.any(ok_b):
        be0, be1 = beta_array[ok_b].min(), beta_array[ok_b].max()
        ax_top.axvspan(be0, be1, color='cyan', alpha=0.25, lw=1)

    # Right: κ_total(M•)
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.scatter(np.log10(M0_array), np.log10(prof_M0),  s = 8, marker = 'o', color = 'white', edgecolor = 'black')
    ax_right.plot(np.log10(M0_array), np.log10(prof_M0), color = 'black',lw = 1)

    ax_right.set_ylim(ax_right.get_ylim()[0], ax_right.get_ylim()[1])
    ax_right.set_yticklabels([])
    ax_right.set_xlabel(r'$\log_{10}\,M0 (M_\odot)$')  # (or A for NSC)
    
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    
    # Put label on the RIGHT, centered, and push it OUTSIDE the axes so it never overlaps.
    ax_right.set_ylabel(r'$\log_{10}\,\min_{\beta}\Sigma\kappa$', rotation=270, va="center")
    ax_right.yaxis.set_label_coords(1.25, 0.5)  # <-- key line: move label further right

    if np.any(ok_m):
        me0, me1 = np.log10([M0_array[ok_m].min(), M0_array[ok_m].max()])
        ax_right.axvspan(me0, me1, color='cyan', alpha=0.25, lw=1)

    # Center: 2D landscape
    ax = fig.add_subplot(gs[1, 0])
    Z = np.log10(np.where(np.isfinite(total) & (total > 0), total, np.nan))
    # Z -=  np.nanmin(Z)
    vmin = np.nanpercentile(Z, 2) if np.isfinite(Z).any() else 0
    vmax = np.nanpercentile(Z, 98) if np.isfinite(Z).any() else 1
    im = ax.imshow(
                    Z.T, origin='lower', cmap='inferno',
                    extent=[beta_array.min(), beta_array.max(), logM0.min(), logM0.max()],
                    aspect='auto', vmin=vmin, vmax=vmax
                )
    ax.set_xlabel(r'$\beta$ (deg)')
    ax.set_ylabel(r'$\log_{10}\,A (M_\odot)/pc$')
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    # cbar.set_label(r'$\log_{10}\,\Sigma_s \kappa_s$')

    # Best point markers + guide lines
    if best_plu is not None:
        bx, bm = best_plu['beta_star'], best_plu['M0_star']
        ax.plot(bx, np.log10(bm), marker='s', ms=7, mfc='none', mec='blue', mew=1.2)
        # guide lines into the marginals
        ax.axvline(bx, ls='--', lw=1.1, c='white', alpha=0.9)
        ax.axhline(np.log10(bm), ls='--', lw=1.1, c='white', alpha=0.9)

        ax_top.axvline(bx, ls='--', lw=1.1, c='red', alpha=0.9)
        ax_right.axvline(np.log10(bm), ls='--', lw=1.1, c='red', alpha=0.9)

    ax_top.grid(alpha=0.25); ax_right.grid(alpha=0.25); ax.grid(alpha=0.15)
    fig.subplots_adjust(right=0.9)

    #plt.show()
    
def summarize_global_beta_param(chi_squared_map, beta_array, p_array, sigma_scale=1.0, pname='param', punits=''):
    """
    Like summarize_global_beta_mbh, but generic for a single kinematic parameter p.
    Returns dict with keys: beta_star, beta_err, p_star, p_err, total_kappa
    """
    total = np.nansum(chi_squared_map, axis=0)  # (n_beta, n_p)
    ib, jp = np.unravel_index(np.nanargmin(total), total.shape)
    beta_star = float(beta_array[ib]); p_star = float(p_array[jp])

    # ~1σ from Δκ=1 profiles
    lo, hi = _profile_bounds_1d(beta_array, total[:, jp], delta=1.0)
    beta_err = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(beta_array, total[:, jp]))
    lo, hi = _profile_bounds_1d(p_array,  total[ib, :],  delta=1.0)
    p_err  = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
              else _curvature_sigma_1d(p_array, total[ib, :]))

    beta_err *= sigma_scale; p_err *= sigma_scale

    print("\n=== Best fit ===")
    print(f"β*  = {beta_star:.2f} ± {beta_err:.2f} deg")
    label = f"{pname} ({punits})" if punits else pname
    print(f"{label} = {p_star:.3g} ± {p_err:.3g}")

    return dict(beta_star=beta_star, beta_err=beta_err, p_star=p_star, p_err=p_err, total_kappa=total)



def plot_chi2_vs_param_global(best_dict, p_array, *, title="", x_label="Parameter", logx=False, logy=True):
    """
    Plot global chi2 vs fitted physical parameter at the best beta.
    best_dict must be the output of summarize_global_beta_param(...).
    """
    if best_dict is None:
        return None

    total = np.asarray(best_dict.get("total_kappa", None), dtype=float)
    if total.ndim != 2:
        return None

    p_array = np.asarray(p_array, dtype=float)
    beta_star = float(best_dict["beta_star"])
    p_star = float(best_dict["p_star"])
    p_err = float(best_dict.get("p_err", np.nan))

    # identify closest beta row
    # total has shape (n_beta, n_p)
    # use minimum over beta if you prefer fully marginalized curve:
    # chi_p = np.nanmin(total, axis=0)
    ib = int(np.nanargmin(np.nanmin(np.abs(total - np.nanmin(total)), axis=1)))
    chi_p = total[ib, :]

    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)
    ax.plot(p_array, np.log10(chi_p), "o-", lw=1., ms=4, c = 'black')
    ax.axvline(p_star, ls="--", lw=1.2, c = 'red')
    if np.isfinite(p_err) and p_err > 0:
        ax.axvspan(max(p_star - p_err, np.nanmin(p_array)), p_star + p_err, alpha=0.15, color = 'red')
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize = 11)
    ax.set_ylabel(r"Log($\chi^2$)", fontsize = 11)
    ax.set_title(title, fontsize = 11)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig








def summarize_global_beta_param2(chi_squared_map, beta_array, p1_array, p2_array,
                                 sigma_scale=1.0, p1name='p1', p1units='',
                                 p2name='p2', p2units=''):
    """
    Find the global best (β*, p1*, p2*) minimizing the total κ across shells.
    chi_squared_map shape: (n_shells, n_beta, n_p1, n_p2)
    """
    total = np.nansum(chi_squared_map, axis=0)  # (NB, NP1, NP2)
    ib, j1, j2 = np.unravel_index(np.nanargmin(total), total.shape)
    beta_star  = float(beta_array[ib])
    p1_star    = float(p1_array[j1])
    p2_star    = float(p2_array[j2])

    # ~1σ from Δκ=1 profiles on 1D marginals at the best point
    # β error from total[:, j1, j2] vs β
    lo, hi = _profile_bounds_1d(beta_array, total[:, j1, j2], delta=1.0)
    beta_err = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
                else _curvature_sigma_1d(beta_array, total[:, j1, j2]))

    # p1 error from total[ib, :, j2] vs p1
    lo, hi = _profile_bounds_1d(p1_array, total[ib, :, j2], delta=1.0)
    p1_err  = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
               else _curvature_sigma_1d(p1_array, total[ib, :, j2]))

    # p2 error from total[ib, j1, :] vs p2
    lo, hi = _profile_bounds_1d(p2_array, total[ib, j1, :], delta=1.0)
    p2_err  = (0.5*(hi-lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) and hi>lo
               else _curvature_sigma_1d(p2_array, total[ib, j1, :]))

    beta_err *= sigma_scale; p1_err *= sigma_scale; p2_err *= sigma_scale

    def _fmt(name, val, err, units):
        lab = f"{name} ({units})" if units else name
        return f"{lab} = {val:.3g} ± {err:.3g}"

    print("\n=== Best fit ===")
    print(f"β*  = {beta_star:.2f} ± {beta_err:.2f} deg")
    print(_fmt(p1name, p1_star, p1_err, p1units))
    print(_fmt(p2name, p2_star, p2_err, p2units))

    return dict(beta_star=beta_star, beta_err=beta_err,
                p1_star=p1_star, p1_err=p1_err,
                p2_star=p2_star, p2_err=p2_err,
                total_kappa=total)



def summarize_keplerian_at_fixed_beta(chi_squared_map, beta_array, v_array, beta_fixed, sigma_scale=1.0):
    """
    Fit a Keplerian MBH at fixed inclination beta_fixed.
    Assumes chi_squared_map has already been computed with FIT_MODE='disk_kepler' style velocities.
    Returns dict with best-fit MBH and diagnostics.
    """
    S, NB, NV = chi_squared_map.shape
    ib = int(np.argmin(np.abs(beta_array - float(beta_fixed))))
    beta_fixed = float(beta_array[ib])

    kappa_sum = np.nansum(chi_squared_map[:, ib, :], axis=0)
    iv_best = int(np.nanargmin(kappa_sum))
    MBH_star = v_array[iv_best]   # if v_array = MBH grid
    kappa_best = np.nanmin(kappa_sum)

    # errors from Δκ=1 around minimum
    lo, hi = _profile_bounds_1d(v_array, kappa_sum, delta=1.0)
    MBH_err = 0.5*(hi - lo)/np.sqrt(2.0) if np.isfinite(lo) and np.isfinite(hi) else np.nan

    print(f"\n=== Keplerian fit at fixed β={beta_fixed:.1f}° ===")
    print(f"Best log10(MBH) = {np.log10(MBH_star):.2f} ± {0.5*(np.log10(hi)-np.log10(lo)) if np.isfinite(lo) and np.isfinite(hi) else np.nan:.2f}")
    print(f"κ_min = {kappa_best:.3f}")

    return dict(
        beta_star=beta_fixed,
        mbh_star=MBH_star,
        mbh_err=MBH_err * sigma_scale,
        kappa_best=kappa_best,
        kappa_profile=kappa_sum
    )



def summarize_fixed_beta_per_shell_v(chi_squared_map, beta_array, v_array,
                                     beta_fixed, sigma_scale=1.0, beta_err_scalar=None):
    """
    Hold β fixed; for each shell pick v*_s = argmin_v κ_s(β_fixed, v)
    and estimate ~1σ errors from Δκ = 1 along v at that β.
    If beta_err_scalar is provided, it will be attached to the output
    (so plots can show the same β uncertainty band you found earlier).
    """
    S, NB, NV = chi_squared_map.shape
    # snap to the nearest β-grid index
    ib = int(np.argmin(np.abs(beta_array - float(beta_fixed))))
    beta_used = float(beta_array[ib])

    v_star = np.full(S, np.nan)
    v_err  = np.full(S, np.nan)

    for s in range(S):
        prof = chi_squared_map[s, ib, :]  # κ_s(v) at fixed β
        if not np.any(np.isfinite(prof)):
            continue
        jv = int(np.nanargmin(prof))
        v_star[s] = float(v_array[jv])

        lo, hi = _profile_bounds_1d(v_array, prof, delta=1.0)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            v_err[s] = 0.5 * (hi - lo) / np.sqrt(2.0)
        else:
            v_err[s] = _curvature_sigma_1d(v_array, prof)

    out = dict(
        beta=np.full(S, beta_used),
        beta_err=np.full(S, np.nan if beta_err_scalar is None else float(beta_err_scalar)),
        v=v_star,
        v_err=v_err * sigma_scale,
        beta_star=beta_used,
        beta_err_scalar=(np.nan if beta_err_scalar is None else float(beta_err_scalar)),
    )
    return out


def vrot_arctan(rad_arcsec, theta, phi, pars):
    """
    Arctangent rotation curve:
      V(r) = (2/pi) * Vmax * arctan(r / rt)

    pars = [Vmax_kms, rt_arcsec]
    Returns V(r) in km/s.
    """
    Vmax_kms, rt_arcsec = pars
    r = np.asarray(rad_arcsec, dtype=float)
    rt = max(float(rt_arcsec), 1e-6)
    return (2.0/np.pi) * float(Vmax_kms) * np.arctan(r/rt)


# old version wrong for bicones
def _shell_edges_from_mask(shape, center_xy, inc_deg, pa_deg,
                            base_mask, r_min_pix, r_max_pix, n_shells,
                            aperture_deg=None, double_cone=False,
                            *, geometry=None):
    """
    Same as your function but geometry is explicit (no global).
    geometry: 'cylindrical' or 'spherical'
    """
    if geometry is None:
        geometry = _ctx_get("geometry", required=True)

    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    xx = xx - center_xy[0]
    yy = yy - center_xy[1]
    xr, yr = _rotate_to_pa(xx, yy, pa_deg)

    if geometry.lower() == 'cylindrical':
        q = np.clip(np.cos(np.radians(inc_deg)), 0.01, 1.0)
    else:
        q = np.clip(np.sin(np.radians(inc_deg)), 0.01, 1.0)

    rell = np.sqrt(xr**2 + (yr / q)**2)

    mask = base_mask.copy()
    if aperture_deg is not None:
        ang_int = np.arctan2((yr / q), xr)
        half = np.radians(aperture_deg) / 2.0
        cone = (np.abs(ang_int) <= half)
        if double_cone:
            opp = (np.abs(((ang_int + np.pi) % (2*np.pi)) - np.pi) <= half)
            cone |= opp
        mask &= cone

    r_obs = rell[mask]
    if r_obs.size == 0 or not np.isfinite(r_obs).any():
        r_lo, r_hi = r_min_pix, r_max_pix
    else:
        r_lo = max(r_min_pix, float(np.nanpercentile(r_obs, 0.0)))
        r_hi = min(r_max_pix, float(np.nanmax(r_obs)))
        if (not np.isfinite(r_lo)) or (not np.isfinite(r_hi)) or (r_hi <= r_lo):
            r_lo, r_hi = r_min_pix, r_max_pix

    return np.linspace(r_lo, r_hi, n_shells + 1)






def residuals_percentiles_cone(
    cube_model, cube_obs, vel_axis, center_xy,
    inc_deg, pa_deg, n_shells,
    r_min_pix, r_max_pix,
    aperture_deg, double_cone,
    perc=(0.01, 0.99),
    sigma_perc_kms=20.0,
    min_pixels_per_shell=10,
    mask_mode="intersection",
    edges_mode="obs",
    perc_weights=1.0,
    *,
    loss="extreme",
    qgrid=None,
    geometry=None,
    FIT_MODE=None,
    KEPLER_DEPROJECT=None
):
    """
    Same math, but now all former globals are explicit (or pulled from km context).
    """
    if geometry is None:
        geometry = _ctx_get("geometry", required=True)
    if FIT_MODE is None:
        FIT_MODE = _ctx_get("FIT_MODE", default="")
    if KEPLER_DEPROJECT is None:
        KEPLER_DEPROJECT = _ctx_get("KEPLER_DEPROJECT", default=False)

    shape      = cube_obs.shape[1:]
    obs_mask   = _finite_mask_from_cube(cube_obs)
    model_mask = _finite_mask_from_cube(cube_model)

    base_mask = _pick_mask(mask_mode,  obs_mask, model_mask)
    edge_mask = _pick_mask(edges_mode, obs_mask, model_mask)

    # restore your legacy PA convention
    pa_math = pa_astro_to_mask_angle(pa_deg, geometry)
    ap_for_edges = None if geometry.lower() == 'cylindrical' else aperture_deg

    r_edges_pix = _shell_edges_from_mask(
        shape, center_xy, inc_deg, pa_math, edge_mask,
        r_min_pix, r_max_pix, n_shells,
        aperture_deg=ap_for_edges, double_cone=double_cone,
        geometry=geometry
    )

    ring_masks = _make_ring_masks_by_geometry(
        geometry, shape, center_xy, inc_deg, pa_math,
        r_edges_pix, aperture_deg, double_cone, base_mask
    )

    valid_shells = np.array([np.count_nonzero(m) >= min_pixels_per_shell for m in ring_masks])

    # quantiles
    if (loss == "crps"):
        q = np.asarray(qgrid if qgrid is not None else np.linspace(0.05, 0.95, 19), float)
        obs_Q = percentile_velocities_from_cube(cube_obs,   vel_axis, ring_masks, q)
        mdl_Q = percentile_velocities_from_cube(cube_model, vel_axis, ring_masks, q)
    else:
        q = np.asarray(perc, float)
        obs_Q = percentile_velocities_from_cube(cube_obs,   vel_axis, ring_masks, q)
        mdl_Q = percentile_velocities_from_cube(cube_model, vel_axis, ring_masks, q)

    # weights per-quantile
    if np.isscalar(perc_weights):
        w_user = float(perc_weights) * np.ones_like(q, dtype=float)
    else:
        w_user = np.asarray(perc_weights, dtype=float)
        if w_user.shape[0] != q.shape[0]:
            raise ValueError("perc_weights must match quantiles in length.")

    # Kepler deprojection scaling (keeps your existing behavior)
    if (geometry.lower() == 'cylindrical') and str(FIT_MODE).startswith('disk_') and bool(KEPLER_DEPROJECT):
        sb = max(np.sin(np.radians(inc_deg)), 1e-3)
        obs_Q = obs_Q / sb
        mdl_Q = mdl_Q / sb
        sigma_eff = float(sigma_perc_kms) / sb
    else:
        sigma_eff = float(sigma_perc_kms)

    # κ per shell
    if loss == "crps":
        dv = (obs_Q - mdl_Q) / sigma_eff
        w_tau = (q * (1.0 - q))
        W = w_user * w_tau
        kappa_shells = np.nansum(np.abs(dv) * W[None, :], axis=1)
    else:
        dif = (obs_Q - mdl_Q) / sigma_eff
        nfinite = np.sum(np.isfinite(dif), axis=1)
        kappa_shells = np.where(
            nfinite == 0, np.inf, np.nansum(w_user[None, :] * (dif**2), axis=1)
        )

    kappa_shells[~valid_shells] = np.inf
    obs_Q[~valid_shells] = np.nan
    mdl_Q[~valid_shells] = np.nan

    return kappa_shells, (obs_Q, mdl_Q, r_edges_pix, ring_masks, valid_shells, q)



def make_mod(beta, vparam, obs=None, gamma_model=None,
             vel3_func_override=None, vel3_pars_override=None,
             ncloud=None):
    """
    Drop-in replacement:
    - You can call make_mod(beta, vparam, obs, gamma_model, ...)
      OR just make_mod(beta, vparam) if obs/gamma_model are already in context.

    Requires context keys (if not passed explicitly):
      geometry, FIT_MODE, obs, xy_AGN, radius_range_model, theta_range, phi_range, zeta_range,
      logradius, vel_sigma, psf_sigma, lsf_sigma, use_seeds, seeds, scale, RT_ARCSEC, scale, npt
    """
    geometry = _ctx_get("geometry", required=True)
    obse = _ctx_get("obs", required=True)

    FIT_MODE = _ctx_get("FIT_MODE", default="")
    use_seeds = _ctx_get("use_seeds", default=False)
    seeds = _ctx_get("seeds", default=None)
    radius_range_model = _ctx_get("radius_range_model", required=True)
    theta_range = _ctx_get("theta_range", required=True)
    phi_range = _ctx_get("phi_range", required=True)
    zeta_range = _ctx_get("zeta_range", required=True)
    logradius = _ctx_get("logradius", default=False)

    vel_sigma = _ctx_get("vel_sigma", default=0.0)
    psf_sigma = _ctx_get("psf_sigma", default=0.0)
    lsf_sigma = _ctx_get("lsf_sigma", default=0.0)

    if obs is None:
        obs = _ctx_get("obs", required=True)
    if gamma_model is None:
        gamma_model = _ctx_get("gamma_model", required=True)

    xy_AGN = _ctx_get("xy_AGN", required=True)

    if ncloud is not None:
        npt = int(ncloud)
    else:
        npt = int(_ctx_get("npt", default=400000))

    g = geometry.lower()

    vel1_func, vel1_pars = None, [0]
    vel2_func, vel2_pars = None, [0]
    vel3_func, vel3_pars = None, [0]

    if g == 'spherical':
        # radial outflow in channel 1
        vel1_func = vout
        vel1_pars = [float(vparam)]

    elif g == 'cylindrical':
        if FIT_MODE == 'disk_kepler':
            scale = float(_ctx_get("scale", required=True))
            vel3_func = vkep_astropy if vel3_func_override is None else vel3_func_override
            vel3_pars = ([float(vparam), scale]
                         if vel3_pars_override is None else list(vel3_pars_override))

        elif FIT_MODE == 'NSC':
            scale = float(_ctx_get("scale", required=True))
            # default R_nsc if you had hardcoded it before
            default_Rnsc = float(_ctx_get("R_nsc_default", default=5.0))
            vel3_func = vnsc_astropy
            vel3_pars = ([float(vparam), scale, default_Rnsc]
                         if vel3_pars_override is None else list(vel3_pars_override))

        elif FIT_MODE == 'Plummer':
            scale = float(_ctx_get("scale", required=True))
            default_a = float(_ctx_get("a_plu_default", default=4.0))
            vel3_func = vplummer_astropy
            vel3_pars = ([float(vparam), scale, default_a]
                         if vel3_pars_override is None else list(vel3_pars_override))

        elif FIT_MODE == 'disk_arctan':
            vel3_func = vrot_arctan if vel3_func_override is None else vel3_func_override
            if vel3_pars_override is None:
                RT_ARCSEC = _ctx_get("RT_ARCSEC", required=True)
                vel3_pars = [float(vparam), float(RT_ARCSEC)]
            else:
                if len(vel3_pars_override) != 2:
                    raise ValueError("For 'disk_arctan', vel3_pars_override must be [Vmax_kms, rt_arcsec].")
                vel3_pars = [float(vel3_pars_override[0]), float(vel3_pars_override[1])]

        else:
            vel3_func = vout
            vel3_pars = [float(vparam)]
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    m = model(
        npt=npt, use_seeds=use_seeds, seeds=seeds, geometry=geometry,
        radius_range=radius_range_model, theta_range=theta_range, phi_range=phi_range, zeta_range=zeta_range,
        logradius=logradius, flux_func=fexpo,
        vel1_func=vel1_func, vel2_func=vel2_func, vel3_func=vel3_func,
        vel_sigma=vel_sigma, psf_sigma=psf_sigma, lsf_sigma=lsf_sigma,
        cube_range=obs.cube['range'], cube_nbins=obs.cube['nbins']
    )

    m.generate_clouds(flux_pars=[1, 2], vel1_pars=vel1_pars, vel2_pars=vel2_pars, vel3_pars=vel3_pars)
    m.observe_clouds(xycenter=xy_AGN, alpha=0, beta=float(beta), gamma=float(gamma_model), vsys=0)

    m.generate_cube()
    # m.kin_maps_cube()
    # cosimo prova cambio
    m.kin_maps()
    
    
    # sempre cosimo prova a aggiungere
    # m.weight_cube(obse.cube['data'])
    # m.generate_cube(weights=m.cube['weights']
    # m.kin_maps_cube(fluxthr=1e-50)

    return m

def build_model(beta, v, *, rt=None, R_nsc=None, a_plu=None):
    """
    Same external behavior as your old build_model, but driven by km context.
    """
    geometry = _ctx_get("geometry", required=True)
    FIT_MODE = _ctx_get("FIT_MODE", default="")
    obs = _ctx_get("obs", required=True)
    gamma_model = _ctx_get("gamma_model", required=True)
    npt = int(_ctx_get("npt", default=400000))
    scale = _ctx_get("scale", default=None)
    scale = _ctx_get("scale", default=None)

    geom = geometry.lower()
    mode = FIT_MODE

    if geom == "cylindrical" and mode == "disk_kepler":
        MBH = float(v)
        return make_mod(beta, MBH, obs=obs, gamma_model=gamma_model, ncloud=npt)

    if geom == "cylindrical" and mode == "NSC":
        if R_nsc is None:
            raise ValueError("NSC mode requires R_nsc (pc).")
        if scale is None:
            raise ValueError("Context missing scale for NSC.")
        A = float(v)
        vel3_pars_nsc = [A, float(scale), float(R_nsc)]
        return make_mod(beta, A, obs=obs, gamma_model=gamma_model,
                        vel3_pars_override=vel3_pars_nsc, ncloud=npt)

    if geom == "cylindrical" and mode == "Plummer":
        if a_plu is None:
            raise ValueError("Plummer mode requires a_plu (pc).")
        if scale is None:
            raise ValueError("Context missing scale for Plummer.")
        M0 = float(v)
        vel3_pars_plu = [M0, float(scale), float(a_plu)]
        return make_mod(beta, M0, obs=obs, gamma_model=gamma_model,
                        vel3_pars_override=vel3_pars_plu, ncloud=npt)

    if geom == "cylindrical" and mode == "disk_arctan":
        if rt is None:
            raise ValueError("disk_arctan requires rt")
        Vmax = float(v)
        return make_mod(beta, Vmax, obs=obs, gamma_model=gamma_model,
                        vel3_pars_override=[Vmax, float(rt)], ncloud=npt)

    # default (independent v)
    return make_mod(beta, float(v), obs=obs, gamma_model=gamma_model, ncloud=npt)


def eval_kappa_for_model(model_cube, beta, pa_deg):
    """
    Uses km context to compare model_cube to the observed cube.
    """
    obs = _ctx_get("obs", required=True)
    vel_axis = _ctx_get("vel_axis", required=True)       # 1D array in km/s
    origin = _ctx_get("origin", required=True)

    num_shells = int(_ctx_get("num_shells", required=True))
    rin_pix = float(_ctx_get("rin_pix", required=True))
    rout_pix = float(_ctx_get("rout_pix", required=True))
    aperture = _ctx_get("aperture", default=None)
    double_cone = bool(_ctx_get("double_cone", default=False))
    SIGMA_PERC_KMS = float(_ctx_get("SIGMA_PERC_KMS", default=20.0))
    perc = _ctx_get("perc", default=(0.01, 0.99))
    perc_weights = _ctx_get("perc_weights", default=1.0)
    loss = _ctx_get("loss", default="extreme")
    CRPS_QGRID = _ctx_get("CRPS_QGRID", default=None)

    kappa_shells, pack = residuals_percentiles_cone(
        cube_model=model_cube,
        cube_obs=obs.cube["data"],
        vel_axis=vel_axis,
        center_xy=origin,
        inc_deg=float(beta),
        pa_deg=float(pa_deg),
        n_shells=num_shells,
        r_min_pix=rin_pix,
        r_max_pix=rout_pix,
        aperture_deg=aperture,
        double_cone=double_cone,
        perc=perc,
        sigma_perc_kms=SIGMA_PERC_KMS,
        mask_mode="model",
        edges_mode="model",
        min_pixels_per_shell=2,
        perc_weights=perc_weights,
        loss=loss,
        qgrid=CRPS_QGRID
    )
    return kappa_shells, pack





def inspect_percentiles_at(beta, v, *, perc=(0.01, 0.99),
                           perc_weights=1.0, sigma_perc_kms=20.0,
                           rt_arcsec=None, loss="extreme", qgrid=None):
    geometry = _ctx_get("geometry", required=True)
    FIT_MODE = _ctx_get("FIT_MODE", default="")
    obs = _ctx_get("obs", required=True)
    gamma_model = _ctx_get("gamma_model", required=True)

    vel_axis = _ctx_get("vel_axis", required=True)
    origin = _ctx_get("origin", required=True)
    arcsec_per_pix = float(_ctx_get("pixscale", required=True))
    num_shells = int(_ctx_get("num_shells", required=True))
    rin_pix = float(_ctx_get("rin_pix", required=True))
    rout_pix = float(_ctx_get("rout_pix", required=True))
    aperture = _ctx_get("aperture", default=None)
    double_cone = bool(_ctx_get("double_cone", default=False))

    # build model
    if (geometry.lower() == "cylindrical") and (FIT_MODE == "disk_arctan"):
        if rt_arcsec is None:
            rt_arcsec = float(_ctx_get("RT_ARCSEC", required=True))
        model_obj = make_mod(
            beta, v, obs=obs, gamma_model=gamma_model,
            vel3_pars_override=[float(v), float(rt_arcsec)]
        )
    else:
        model_obj = make_mod(beta, v, obs=obs, gamma_model=gamma_model)

    kappa, pack = residuals_percentiles_cone(
        cube_model=model_obj.cube["data"],
        cube_obs=obs.cube["data"],
        vel_axis=vel_axis,
        center_xy=origin,
        inc_deg=float(beta),
        pa_deg=float(gamma_model),
        n_shells=num_shells,
        r_min_pix=rin_pix,
        r_max_pix=rout_pix,
        aperture_deg=aperture,
        double_cone=double_cone,
        perc=perc,
        sigma_perc_kms=float(sigma_perc_kms),
        mask_mode="model",
        edges_mode="model",
        min_pixels_per_shell=2,
        perc_weights=perc_weights,
        loss=loss,
        qgrid=qgrid,
    )

    if not isinstance(pack, tuple) or len(pack) < 2:
        raise TypeError(
            f"inspect_percentiles_at: unexpected pack format. "
            f"Expected tuple with at least 2 elements, got {type(pack)}"
        )

    obs_perc = np.asarray(pack[0], dtype=float)
    mod_perc = np.asarray(pack[1], dtype=float)

    if obs_perc.ndim != 2 or obs_perc.shape[1] != 2:
        raise ValueError(
            f"inspect_percentiles_at: expected obs percentile array with shape (N,2), "
            f"got {obs_perc.shape}"
        )

    if mod_perc.ndim != 2 or mod_perc.shape[1] != 2:
        raise ValueError(
            f"inspect_percentiles_at: expected model percentile array with shape (N,2), "
            f"got {mod_perc.shape}"
        )

    # Prefer shell edges returned by residuals_percentiles_cone
    if len(pack) >= 3:
        edges_pix = np.asarray(pack[2], dtype=float)
        if edges_pix.ndim == 1 and len(edges_pix) >= 2:
            rmid_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
        else:
            edges_pix = np.linspace(float(rin_pix), float(rout_pix), int(num_shells) + 1)
            rmid_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
    else:
        edges_pix = np.linspace(float(rin_pix), float(rout_pix), int(num_shells) + 1)
        rmid_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])

    # valid shell mask if provided
    valid_shells = None
    if len(pack) >= 5:
        try:
            valid_shells = np.asarray(pack[4], dtype=bool)
        except Exception:
            valid_shells = None

    n = min(len(rmid_pix), len(obs_perc), len(mod_perc))
    rr_pix = np.asarray(rmid_pix[:n], dtype=float)
    rr = rr_pix * arcsec_per_pix
    obs_lo = np.asarray(obs_perc[:n, 0], dtype=float)
    obs_hi = np.asarray(obs_perc[:n, 1], dtype=float)
    mod_lo = np.asarray(mod_perc[:n, 0], dtype=float)
    mod_hi = np.asarray(mod_perc[:n, 1], dtype=float)

    good = (
        np.isfinite(rr) &
        np.isfinite(obs_lo) & np.isfinite(obs_hi) &
        np.isfinite(mod_lo) & np.isfinite(mod_hi)
    )

    if valid_shells is not None:
        good &= valid_shells[:n]

    if not np.any(good):
        raise ValueError("inspect_percentiles_at: no valid shells available for percentile plotting.")

    # use returned percentiles for legend if available
    perc_used = perc
    if len(pack) >= 6:
        try:
            perc_used_arr = np.asarray(pack[5], dtype=float).ravel()
            if len(perc_used_arr) >= 2:
                perc_used = (float(perc_used_arr[0]), float(perc_used_arr[1]))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

    c_lo = "C0"
    c_hi = "C1"
    ms = 5
    lw = 1.

    ax.plot(
        rr[good], obs_lo[good],
        "o-",
        color=c_lo,
        ms=ms, lw=lw,
        mfc=c_lo, mec=c_lo, mew=1,
        label=f"Obs {perc_used[0]*100:.0f}%"
    )

    ax.plot(
        rr[good], obs_hi[good],
        "o-",
        color=c_hi,
        ms=ms, lw=lw,
        mfc=c_hi, mec=c_hi, mew=1,
        label=f"Obs {perc_used[1]*100:.0f}%"
    )

    ax.plot(
        rr[good], mod_lo[good],
        "o-",
        color=c_lo,
        ms=ms, lw=lw,
        mfc="none", mec=c_lo, mew=1,
        label=f"Mod {perc_used[0]*100:.0f}%"
    )

    ax.plot(
        rr[good], mod_hi[good],
        "o-",
        color=c_hi,
        ms=ms, lw=lw,
        mfc="none", mec=c_hi, mew=1,
        label=f"Mod {perc_used[1]*100:.0f}%"
    )


    ax.set_xlabel("Radius [arcsec]", fontsize=11)
    ax.set_ylabel(r"Velocity [km s$^{-1}$]", fontsize = 11)
    ax.set_title("Percentile comparison", fontsize = 11)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10, loc="best", ncol = 2)
    ax.tick_params(axis='both', direction='in', labelsize=11, width=2)

    plt.tight_layout()

    return fig, kappa, pack




################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
#%% Function for the Novello fitting part only

def _describe_missing_pixscale_header_info(obshead) -> str:
    """
    Build a human-readable message describing which WCS/header keywords
    relevant for pixel scale are missing.
    """
    keys_to_check = [
        "CTYPE1", "CTYPE2",
        "CUNIT1", "CUNIT2",
        "CDELT1", "CDELT2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    ]

    missing = []
    present = []

    for k in keys_to_check:
        if k in obshead and obshead[k] not in (None, ""):
            present.append(f"{k}={obshead[k]}")
        else:
            missing.append(k)

    msg = []
    msg.append("Pixel scale could not be determined from the FITS header/WCS.")
    msg.append("Relevant spatial-WCS keywords present: " + (", ".join(present) if present else "none"))
    msg.append("Relevant spatial-WCS keywords missing: " + (", ".join(missing) if missing else "none"))

    return "\n".join(msg)

def apply_spatial_mask_to_cube(cube_spec_yx, mask_yx, mode="zero"):
    """
    mode:
      - "zero": set masked pixels to 0
      - "nan":  set masked pixels to NaN
    """
    out = np.array(cube_spec_yx, copy=True)
    if mode == "zero":
        out[:, mask_yx] = 0.0
    elif mode == "nan":
        out[:, mask_yx] = np.nan
    else:
        raise ValueError("mode must be 'zero' or 'nan'")
    return out



def make_observed_like(obs_template, cube_spec_yx, fluxmap=None, velmap=None, sigmap=None):
    """
    Build a observed instance consistent with an existing one.

    Some observed objects don't store crval/cdelt/crpix inside obs_template.cube,
    so we read them from attributes if present, otherwise from obs_template.cube.
    """

    # --- robust getters ---
    def _get_any(obj, names, cube_keys):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        for k in cube_keys:
            if isinstance(obj.cube, dict) and (k in obj.cube):
                return obj.cube[k]
        raise KeyError(f"Could not find {names} / {cube_keys} in observed template.")

    crval = _get_any(obs_template, names=("crval",), cube_keys=("crval",))
    cdelt = _get_any(obs_template, names=("cdelt",), cube_keys=("cdelt",))
    crpix = _get_any(obs_template, names=("crpix",), cube_keys=("crpix",))

    o = observed(cube_spec_yx, error=None, crval=crval, cdelt=cdelt, crpix=crpix,
                 fluxmap=fluxmap, velmap=velmap, sigmap=sigmap)

    # copy cached grids used by kin_maps()
    for attr in ("xobs", "yobs", "xobs_psf", "yobs_psf"):
        if hasattr(obs_template, attr):
            setattr(o, attr, getattr(obs_template, attr))

    return o


def fit_gridsearch_component(
    *,
    obs_for_fit,
    disc_cube=None,
    vel_axis,
    origin,
    pixscale,
    nrebin,
    scale,
    # model / context
    geometry,                 # "cylindrical" or "spherical"
    FIT_MODE,
    gamma_model_deg,          # PA for this component (disc PA or outflow axis PA)
    aperture_deg,
    double_cone,
    radius_range_model_arcsec,
    theta_range,
    phi_range,
    zeta_range,
    logradius,
    psf_sigma,
    lsf_sigma,
    vel_sigma,
    npt,
    # fitting metric
    num_shells,
    perc,
    perc_weights,
    loss,                     # "extreme" or "crps"
    CRPS_QGRID,
    SIGMA_PERC_KMS,
    # grid ranges
    beta_min, beta_max, step_beta,
    v_min, v_max, step_v,
    # extra
    R_nsc=5.0,
    a_plu=4.0,
    RT_ARCSEC=None,
    n_geom_v=50,
    verbose_label="Fit"
):
    """
    Generic gridsearch wrapper that uses your existing:
      - set_fit_context
      - build_model(beta, v, ...)
      - eval_kappa_for_model
      - summarize_global_beta_with_per_shell_v (default summary)

    Returns dict with chi map + best parameters.
    """
    arcsec_per_pix = float(pixscale) * float(nrebin)
    rin_pix  = int(round(float(radius_range_model_arcsec[0]) / arcsec_per_pix))
    rout_pix = int(round(float(radius_range_model_arcsec[1]) / arcsec_per_pix))

    beta_array = np.arange(beta_min, beta_max + step_beta, step_beta)

    # weights handling (your convention)
    pw = 1.0 if (loss == "crps") else perc_weights

    set_fit_context(
        geometry=geometry,
        FIT_MODE=FIT_MODE,
        KEPLER_DEPROJECT=False,
        disc_cube=disc_cube,
        obs=obs_for_fit,
        vel_axis=vel_axis,
        origin=origin,
        pixscale = pixscale,
        xy_AGN=[0.0, 0.0],
        gamma_model=float(gamma_model_deg),
        num_shells=num_shells,
        rin_pix=rin_pix,
        rout_pix=rout_pix,
        aperture=float(aperture_deg),
        double_cone=bool(double_cone),
        SIGMA_PERC_KMS=float(SIGMA_PERC_KMS),
        perc=perc,
        perc_weights=pw,
        loss=loss,
        CRPS_QGRID=CRPS_QGRID,
        scale=scale,
        RT_ARCSEC=RT_ARCSEC,
        npt=int(npt),
        radius_range_model=list(radius_range_model_arcsec),
        theta_range=theta_range,
        phi_range=phi_range,
        zeta_range=zeta_range,
        logradius=bool(logradius),
        psf_sigma=psf_sigma,
        lsf_sigma=float(lsf_sigma),
        vel_sigma=float(vel_sigma),
        v_min=float(v_min),
        v_max=float(v_max),
        step_v=float(step_v),
        R_nsc_default=float(R_nsc) if R_nsc is not None else 5.0,
        a_plu_default=float(a_plu) if a_plu is not None else 4.0,
        n_geom_v=int(n_geom_v),
    )
    
    v_array, rt_array, Y_LABEL = build_v_grid_and_label(geometry, FIT_MODE)


    num_beta, num_v = len(beta_array), len(v_array)
    chi_squared_map = np.zeros((num_shells, num_beta, num_v), dtype=float)

    tot_it = num_beta * num_v
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>6.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        ) as progress:
        task = progress.add_task(f"{verbose_label} grid search", total=tot_it)
        for ib, beta in enumerate(beta_array):
            for iv, v in enumerate(v_array):

                # your existing computation here
                # -------------------------------

                progress.advance(task)

                disc_cube = _ctx_get("disc_cube", default=None)

                # ---- primary lobe ----
                model1 = build_model(beta, v, rt=RT_ARCSEC, R_nsc=R_nsc, a_plu=a_plu)
            
                # ---- second lobe for bicones ----
                if bool(double_cone):
                    beta_flip  = 180.0 - beta
                    gamma_flip = (gamma_model_deg + 180.0) % 360.0
            
                    # temporarily override gamma
                    _FIT_CTX["gamma_model"] = gamma_flip
                    model2 = build_model(beta_flip, v, rt=RT_ARCSEC, R_nsc=R_nsc, a_plu=a_plu)
                    _FIT_CTX["gamma_model"] = gamma_model_deg      # restore
            
                    combined_model_cube = disc_cube + model1.cube["data"] + model2.cube["data"]
                else:
                    combined_model_cube = disc_cube + model1.cube["data"] if disc_cube is not None else model1.cube["data"]
            
                # ---- evaluate χ² ----
                kappa_shells, _ = eval_kappa_for_model(combined_model_cube, beta, float(gamma_model_deg))

                chi_squared_map[:, ib, iv] = kappa_shells
 
    print(f"\n{verbose_label} grid evaluation completed")

    # Default summary
    mode = str(FIT_MODE)
    geom = str(geometry).lower()

    if geom == "cylindrical" and mode in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}:
        best = summarize_global_beta_param(
            chi_squared_map,
            beta_array,
            v_array,
            sigma_scale=1.0,
            pname={
                "disk_kepler": "M_BH",
                "NSC": "A",
                "Plummer": "M0",
                "disk_arctan": "Vmax",
            }[mode],
            punits={
                "disk_kepler": "Msun",
                "NSC": "",
                "Plummer": "Msun",
                "disk_arctan": "km/s",
            }[mode],
        )
        beta_best = float(best["beta_star"])
        v_best = float(best["p_star"])
    else:
        best = summarize_global_beta_with_per_shell_v(chi_squared_map, beta_array, v_array)
        beta_best = float(best["beta_star"])
        v_best = float(np.nanmedian(best["v"])) if np.isfinite(best.get("v", np.nan)).any() else float(v_array[0])


    return dict(
        chi_squared_map=chi_squared_map,
        beta_array=beta_array,
        v_array=v_array,
        best=best,
        beta_best=beta_best,
        v_best=v_best,
        rin_pix=rin_pix,
        rout_pix=rout_pix,
        aperture=float(aperture_deg),
        gamma_model=float(gamma_model_deg),
        geometry=str(geometry),
        FIT_MODE=str(FIT_MODE),
        double_cone=bool(double_cone),
        radius_range_model_arcsec=list(radius_range_model_arcsec)
    )


def build_best_model_from_fit(*, beta_best, v_best, FIT_MODE, R_nsc=5.0, a_plu=4.0, rt=None):
    """
    Convenience: build a model with your existing builder using the best params.
    """
    return build_model(beta_best, v_best, rt=rt if (FIT_MODE == "disk_arctan") else None,
                       R_nsc=R_nsc, a_plu=a_plu)



def _bicone_masks(shape, center_xy, beta1, gamma1, beta2, gamma2, 
                  aperture_deg, r_edges_pix, obs_mask):
    """
    Build combined masks for a true bicone with different geometry for each lobe.
    """
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    xx = xx - center_xy[0]
    yy = yy - center_xy[1]
    
    def _single_cone_mask_and_radius(beta, gamma_math):
        xr, yr = _rotate_to_pa(xx, yy, gamma_math)
        
        # project with cone's inclination
        q = np.clip(np.sin(np.radians(beta)), 0.01, 1.0)
        rell = np.sqrt(xr**2 + (yr / q)**2)
        
        # intrinsic angle in cone coordinates
        ang_int = np.arctan2((yr / q), xr)
        
        # aperture mask
        half = np.radians(aperture_deg) * 0.5
        cone_mask = (np.abs(ang_int) <= half)
        
        return cone_mask, rell
    
    # get masks and radii for both lobes
    cone1, rell1 = _single_cone_mask_and_radius(beta1, gamma1)
    cone2, rell2 = _single_cone_mask_and_radius(beta2, gamma2)
    
    # combine: use union of cone regions
    combined_cone = (cone1 | cone2) & obs_mask
    
    # for radius: use the appropriate radius for each region
    rell_combined = np.where(cone1, rell1, np.where(cone2, rell2, np.inf))
    
    # build shell masks
    masks = []
    for i in range(len(r_edges_pix) - 1):
        ring = (rell_combined >= r_edges_pix[i]) & (rell_combined < r_edges_pix[i + 1])
        masks.append(ring & combined_cone)
    
    return masks



def summarize_free_beta_per_shell(chi2_cube, beta_array, v_array, delta_chi2=2.30):
    """
    For each shell, find the (beta, v) pair that minimizes chi2.
    Uncertainties are estimated from the region chi2 <= chi2_min + delta_chi2
    (default 2.30 ~ 1σ for 2 parameters).

    Returns dict with beta, beta_err, v, v_err arrays (length = n_shells).
    """
    chi2_cube = np.asarray(chi2_cube, float)              # (shell, nbeta, nv)
    beta_array = np.asarray(beta_array, float)            # (nbeta,)
    v_array = np.asarray(v_array, float)                  # (nv,)

    ns = chi2_cube.shape[0]
    beta_best = np.full(ns, np.nan)
    v_best    = np.full(ns, np.nan)
    beta_err  = np.full(ns, np.nan)
    v_err     = np.full(ns, np.nan)

    for i in range(ns):
        chi2 = chi2_cube[i]
        if not np.any(np.isfinite(chi2)):
            continue

        j, k = np.unravel_index(np.nanargmin(chi2), chi2.shape)
        beta_best[i] = beta_array[j]
        v_best[i]    = v_array[k]

        chi2min = chi2[j, k]
        if not np.isfinite(chi2min):
            continue

        region = np.isfinite(chi2) & (chi2 <= chi2min + float(delta_chi2))
        if not np.any(region):
            continue

        jj, kk = np.where(region)
        bvals = beta_array[jj]
        vvals = v_array[kk]

        # symmetric half-width errors
        beta_err[i] = 0.5 * (np.nanmax(bvals) - np.nanmin(bvals))
        v_err[i]    = 0.5 * (np.nanmax(vvals) - np.nanmin(vvals))

    return {
        "beta": beta_best,
        "beta_err": beta_err,
        "v": v_best,
        "v_err": v_err,
    }






################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
#%% Additional functions for the novello fit


def _get_disc_global_param_best(disc_fit, disc_fit_mode):
    if disc_fit_mode in {"disk_kepler", "NSC", "Plummer", "disk_arctan"}:
        best = disc_fit.get("best", {})
        return float(best.get("p_star", disc_fit.get("v_best", np.nan)))
    return float(disc_fit.get("v_best", np.nan))






def _extract_best_fit_with_uncertainties(fit: dict | None) -> dict | None:
    """
    Extract best-fit beta/v-like quantity and approximate uncertainties from the fit dictionary.

    For standard fits:
        returns beta_best, beta_err, v_best, v_err

    For physical disc fits:
        v_best / v_err contain the fitted global parameter (MBH, A, M0, or Vmax),
        so the caller can still log them generically.
    """
    if fit is None:
        return None

    beta_best = float(fit.get("beta_best", np.nan))
    v_best = float(fit.get("v_best", np.nan))

    beta_err = np.nan
    v_err = np.nan

    best = fit.get("best", None)
    if isinstance(best, dict):
        # physical/global-parameter summary
        if "p_star" in best:
            beta_err = float(best.get("beta_err", np.nan))
            v_best = float(best.get("p_star", v_best))
            v_err = float(best.get("p_err", np.nan))

        # old per-shell/global-beta summary
        else:
            beta_err = float(best.get("beta_err_scalar", np.nan))

            v_arr = np.asarray(best.get("v", []), dtype=float)
            v_err_arr = np.asarray(best.get("v_err", []), dtype=float)

            if v_arr.size > 0 and v_err_arr.size > 0 and np.isfinite(v_best):
                idx = int(np.nanargmin(np.abs(v_arr - v_best)))
                if idx < v_err_arr.size:
                    v_err = float(v_err_arr[idx])

    return {
        "beta_best": beta_best,
        "beta_err": beta_err,
        "v_best": v_best,
        "v_err": v_err,
    }




def _shell_midpoints_and_halfwidths_arcsec(rin_pix, rout_pix, n_shells, arcsec_per_pix):
    edges_pix = np.linspace(float(rin_pix), float(rout_pix), int(n_shells) + 1)
    edges_arc = edges_pix * arcsec_per_pix
    rmid_arc = 0.5 * (edges_arc[:-1] + edges_arc[1:])
    xerr_arc = 0.5 * (edges_arc[1:] - edges_arc[:-1])
    return rmid_arc, xerr_arc

def _escape_factor_from_eta(eta):
    eta = np.asarray(eta, dtype=float)
    out = np.full_like(eta, np.nan, dtype=float)
    good = np.isfinite(eta) & (eta > 1.0)
    out[good] = np.sqrt(2.0 * (1.0 + np.log(eta[good])))
    return out




def _ratio_to_escape_and_uncertainty(v_out, e_out, v_c_outer, e_c_outer, eta):
    """
    Compute v_out / v_esc with
        v_esc = sqrt[2 * v_c_outer^2 * (1 + ln(eta))]
              = f_eta * v_c_outer

    Parameters
    ----------
    v_out, e_out : array-like
        Outflow velocity and uncertainty per shell.
    v_c_outer, e_c_outer : float
        Representative outer circular velocity and its uncertainty.
    eta : float or array-like
        r_max / r ratio. Can be scalar (e.g. 10, 30, 100) or one value per shell.

    Returns
    -------
    ratio, ratio_err, v_esc
    """
    v_out = np.asarray(v_out, dtype=float)
    e_out = np.asarray(e_out, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if eta.ndim == 0:
        eta = np.full_like(v_out, float(eta), dtype=float)

    f_eta = _escape_factor_from_eta(eta)
    v_esc = f_eta * float(v_c_outer)
    e_esc = f_eta * float(e_c_outer)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = v_out / v_esc

        frac_out = np.divide(
            e_out, v_out,
            out=np.full_like(e_out, np.nan, dtype=float),
            where=np.isfinite(v_out) & (v_out != 0)
        )
        frac_esc = np.divide(
            e_esc, v_esc,
            out=np.full_like(v_esc, np.nan, dtype=float),
            where=np.isfinite(v_esc) & (v_esc != 0)
        )

        ratio_err = np.abs(ratio) * np.sqrt(frac_out**2 + frac_esc**2)

    bad = (~np.isfinite(ratio)) | (~np.isfinite(ratio_err)) | (~np.isfinite(v_esc))
    ratio[bad] = np.nan
    ratio_err[bad] = np.nan
    v_esc[bad] = np.nan

    return ratio, ratio_err, v_esc




def _extract_radial_profile(best_profile, rin_pix, rout_pix, n_shells, arcsec_per_pix):
    if best_profile is None:
        return None

    v = np.asarray(best_profile.get("v", []), dtype=float)
    if v.size == 0:
        return None

    v_err = np.asarray(best_profile.get("v_err", np.full_like(v, np.nan)), dtype=float)

    rmid_arc, xerr_arc = _shell_midpoints_and_halfwidths_arcsec(
        rin_pix=rin_pix,
        rout_pix=rout_pix,
        n_shells=n_shells,
        arcsec_per_pix=arcsec_per_pix,
    )

    n = min(len(rmid_arc), len(v), len(v_err), len(xerr_arc))

    return {
        "r_arcsec": rmid_arc[:n],
        "xerr_arcsec": xerr_arc[:n],
        "v": v[:n],
        "v_err": v_err[:n],
    }

def flux_unit_scale_from_bunit(bunit_str):
    """
    Return the multiplicative scale factor that converts the cube values into
    cgs flux-density units:

        erg s^-1 cm^-2 Angstrom^-1

    Example:
        '10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)'  -> 1e-20

    Returns
    -------
    scale : float or None
        Multiplicative factor to apply to cube values.
        Returns None if the string cannot be parsed safely.
    """
    if bunit_str is None:
        return None

    s = str(bunit_str).strip().lower()
    if not s:
        return None

    # normalize a few common textual variants
    s = s.replace("angstrom", "Angstrom")
    s = s.replace("ang", "Angstrom")
    s = s.replace("erg", "erg")
    s = s.replace(".s", " s")
    s = s.replace(".cm", " cm")
    s = s.replace(".Angstrom", " Angstrom")
    s = s.replace("*", "*")

    # ---------------------------------------------------------
    # 1) Try to extract an explicit leading factor like 10**(-20)
    # ---------------------------------------------------------
    scale = 1.0
    m = re.search(r"10\*\*\(\s*([+-]?\d+)\s*\)", s)
    if m:
        scale = 10.0 ** float(m.group(1))
        s = re.sub(r"10\*\*\(\s*[+-]?\d+\s*\)", "", s).strip()

    # Remove separators sometimes used in headers
    s = s.replace(".", " ")
    s = " ".join(s.split())

    # ---------------------------------------------------------
    # 2) Parse the remaining physical unit
    # ---------------------------------------------------------
    try:
        unit = u.Unit(s)
    except Exception:
        return None

    target = u.erg / u.s / (u.cm**2) / u.AA

    try:
        conv = (1.0 * unit).to(target).value
    except Exception:
        return None

    return float(scale * conv)


def identify_emission_line(wavelength_line, wavelength_line_unit="Angstrom", tol_angstrom=5.0):
    """
    Identify which standard optical line is being analyzed from the user YAML.
    Returns one of:
        'oiii5007', 'halpha', 'hbeta', or None if unsupported.
    """
    wl = (float(wavelength_line) * u.Unit(wavelength_line_unit)).to(u.AA).value

    if abs(wl - 5006.8) <= tol_angstrom:
        return "oiii5007"
    if abs(wl - 6562.8) <= tol_angstrom:
        return "halpha"
    if abs(wl - 4861.3) <= tol_angstrom:
        return "hbeta"

    return None


def energetics_line_description(line_id):
    """
    Human-readable description of the ionized-gas mass prescription used
    for the energetics calculation.
    """
    if line_id == "oiii5007":
        return (
            "Since you are using the [OIII]5007 emission line to compute the mass, "
            "the code uses an [OIII]-based ionized-gas mass estimate following the "
            "standard Carniani et al.2015 style prescription. "
            "This is more assumption-dependent than Balmer-line estimates because it "
            "depends on ionization and metallicity assumptions."
        )

    if line_id == "halpha":
        return (
            "Since you are using the Halpha emission line to compute the mass, "
            "the code uses the standard recombination-line ionized-gas mass estimate "
            "based on Halpha luminosity under case-B assumptions."
        )

    if line_id == "hbeta":
        return (
            "Since you are using the Hbeta emission line to compute the mass, "
            "the code uses the standard recombination-line ionized-gas mass estimate "
            "based on Hbeta luminosity under case-B assumptions."
        )

    return (
        "Energetics are currently implemented only for [OIII]5007, Halpha, and Hbeta. "
        "For other emission lines this part of the analysis is still work in progress."
    )


def emission_line_label(line_id):
    if line_id == "oiii5007":
        return "[OIII]5007"
    if line_id == "halpha":
        return "Halpha"
    if line_id == "hbeta":
        return "Hbeta"
    return "unsupported"


def load_ne_map(ne_map_path):
    """
    Read a density map from a FITS file.
    Requirement: single 2D image in one extension (primary or first image extension).
    """
    if ne_map_path is None:
        return None

    with fits.open(ne_map_path) as hdul:
        # prefer first HDU with 2D data
        data2d = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) == 2:
                data2d = np.array(hdu.data, dtype=float, copy=True)
                break

    if data2d is None:
        raise ValueError(f"No 2D density map found in FITS file: {ne_map_path}")

    data2d[~np.isfinite(data2d)] = np.nan
    data2d[data2d <= 0] = np.nan
    return data2d


def radial_shell_edges_pix(rmin_pix, rmax_pix, n_shells):
    return np.linspace(float(rmin_pix), float(rmax_pix), int(n_shells) + 1)


def radial_shell_masks_yx(shape_yx, center_xy, r_edges_pix, extra_mask=None):
    """
    Circular annular shell masks on the sky plane.
    For outflow energetics this is fine because the lobe masking is already applied
    outside this function (obs_out_pos / obs_out_neg cubes).
    """
    ny, nx = shape_yx
    x0, y0 = center_xy
    yy, xx = np.indices((ny, nx), dtype=float)
    rr = np.sqrt((xx - float(x0))**2 + (yy - float(y0))**2)

    masks = []
    for i in range(len(r_edges_pix) - 1):
        mm = (rr >= r_edges_pix[i]) & (rr < r_edges_pix[i + 1])
        if extra_mask is not None:
            mm = mm & extra_mask
        masks.append(mm)
    return masks


def integrated_line_flux_per_shell_from_cube(
    cube_data,
    shell_masks,
    dv_kms,
    lambda_obs_angstrom,
    flux_unit_scale=1.0,
)   :
    """
    Compute integrated line flux in each shell from the cube.

    IMPORTANT:
    This assumes the cube is in flux density units per Angstrom,
    so F_line = sum(cube) * d_lambda.

    d_lambda = lambda_obs * dv / c
    """
    cube_data = np.asarray(cube_data, dtype=float)
    dlam = float(lambda_obs_angstrom) * abs(float(dv_kms)) / 299792.458

    flux_shell = np.full(len(shell_masks), np.nan, dtype=float)
    npix_shell = np.zeros(len(shell_masks), dtype=int)

    for i, mask2d in enumerate(shell_masks):
        npix_shell[i] = int(np.count_nonzero(mask2d))
        if npix_shell[i] == 0:
            continue

        shell_sum = np.nansum(cube_data[:, mask2d]) * float(flux_unit_scale)
        flux_shell[i] = shell_sum * dlam

    return flux_shell, npix_shell


def shell_density_from_map(ne_map, shell_masks, reducer="median"):
    """
    Derive one n_e value per shell from a 2D density map.
    Default is median for robustness.
    """
    ne_shell = np.full(len(shell_masks), np.nan, dtype=float)

    for i, mask2d in enumerate(shell_masks):
        vals = np.asarray(ne_map[mask2d], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue

        if reducer == "mean":
            ne_shell[i] = np.nanmean(vals)
        else:
            ne_shell[i] = np.nanmedian(vals)

    return ne_shell


def luminosity_from_flux(flux_erg_s_cm2, luminosity_distance_mpc):
    """
    L = 4 pi D_L^2 F
    """
    dl_cm = (float(luminosity_distance_mpc) * u.Mpc).to(u.cm).value
    return 4.0 * np.pi * dl_cm**2 * np.asarray(flux_erg_s_cm2, dtype=float)


def ionized_mass_from_luminosity(line_id, luminosity_erg_s, ne_cm3, z_over_zsun=1.0):
    """
    Standard ionized-gas mass scalings.

    Hbeta:
        M ~ 7.2e8 (L_Hb / 1e43) (100 / ne) Msun

    Halpha:
        M ~ 2.5e8 (L_Ha / 1e43) (100 / ne) Msun

    [OIII]5007:
        M ~ 4.0e7 (L_OIII / 1e44) (1000 / ne) (1 / Z) Msun
        with Z = z_over_zsun * Zsun

    Returns mass in Msun.
    """
    L = np.asarray(luminosity_erg_s, dtype=float)
    ne = np.asarray(ne_cm3, dtype=float)

    out = np.full(np.broadcast(L, ne).shape, np.nan, dtype=float)
    good = np.isfinite(L) & np.isfinite(ne) & (L > 0) & (ne > 0)

    if not np.any(good):
        return out

    if line_id == "hbeta":
        out[good] = 7.2e8 * (L[good] / 1.0e43) * (100.0 / ne[good])

    elif line_id == "halpha":
        out[good] = 2.5e8 * (L[good] / 1.0e43) * (100.0 / ne[good])

    elif line_id == "oiii5007":
        zfac = float(z_over_zsun) if np.isfinite(z_over_zsun) and (z_over_zsun > 0) else 1.0
        out[good] = 4.0e7 * (L[good] / 1.0e44) * (1000.0 / ne[good]) * (1.0 / zfac)

    else:
        raise ValueError(f"Unsupported line_id={line_id}")

    return out


def mass_outflow_rate_msun_per_yr(mass_msun, velocity_kms, delta_r_kpc):
    """
    dot(M) = M * v / DeltaR
    """
    M = np.asarray(mass_msun, dtype=float) * u.Msun
    v = np.asarray(velocity_kms, dtype=float) * (u.km / u.s)
    dR = np.asarray(delta_r_kpc, dtype=float) * u.kpc

    out = np.full(np.broadcast(np.asarray(mass_msun, float),
                               np.asarray(velocity_kms, float),
                               np.asarray(delta_r_kpc, float)).shape,
                  np.nan, dtype=float)

    good = np.isfinite(M.value) & np.isfinite(v.value) & np.isfinite(dR.value) & (M.value > 0) & (v.value > 0) & (dR.value > 0)
    if np.any(good):
        out[good] = ((M[good] * v[good] / dR[good]).to(u.Msun / u.yr)).value
    return out


def kinetic_power_erg_s(mdot_msun_yr, velocity_kms):
    mdot = np.asarray(mdot_msun_yr, dtype=float) * (u.Msun / u.yr)
    v = np.asarray(velocity_kms, dtype=float) * (u.km / u.s)

    out = np.full(np.broadcast(np.asarray(mdot_msun_yr, float),
                               np.asarray(velocity_kms, float)).shape,
                  np.nan, dtype=float)

    good = np.isfinite(mdot.value) & np.isfinite(v.value) & (mdot.value > 0) & (v.value > 0)
    if np.any(good):
        out[good] = (0.5 * mdot[good].to(u.g / u.s) * v[good].to(u.cm / u.s)**2).to(u.erg / u.s).value
    return out


def momentum_rate_dyne(mdot_msun_yr, velocity_kms):
    mdot = np.asarray(mdot_msun_yr, dtype=float) * (u.Msun / u.yr)
    v = np.asarray(velocity_kms, dtype=float) * (u.km / u.s)

    out = np.full(np.broadcast(np.asarray(mdot_msun_yr, float),
                               np.asarray(velocity_kms, float)).shape,
                  np.nan, dtype=float)

    good = np.isfinite(mdot.value) & np.isfinite(v.value) & (mdot.value > 0) & (v.value > 0)
    if np.any(good):
        out[good] = (mdot[good].to(u.g / u.s) * v[good].to(u.cm / u.s)).to(u.dyne).value
    return out


def build_outflow_energetics_profile(
    *,
    cube_data,
    center_xy,
    rmin_pix,
    rmax_pix,
    n_shells,
    arcsec_per_pix,
    scale_kpc_per_arcsec,
    dv_kms,
    lambda_obs_angstrom,
    luminosity_distance_mpc,
    velocity_profile,
    ne_shell,
    line_id,
    z_over_zsun=1.0,
    extra_mask=None,
    flux_unit_scale=1.0,
):
    """
    Build a shell-by-shell energetics profile for one lobe.
    """
    r_edges_pix = radial_shell_edges_pix(rmin_pix, rmax_pix, n_shells)
    shell_masks = radial_shell_masks_yx(
        shape_yx=cube_data.shape[1:],
        center_xy=center_xy,
        r_edges_pix=r_edges_pix,
        extra_mask=extra_mask,
    )

    flux_shell, npix_shell = integrated_line_flux_per_shell_from_cube(
        cube_data=cube_data,
        shell_masks=shell_masks,
        dv_kms=dv_kms,
        lambda_obs_angstrom=lambda_obs_angstrom,
        flux_unit_scale=flux_unit_scale,
    )

    lum_shell = luminosity_from_flux(flux_shell, luminosity_distance_mpc)

    rmid_pix = 0.5 * (r_edges_pix[:-1] + r_edges_pix[1:])
    dr_pix = (r_edges_pix[1:] - r_edges_pix[:-1])

    rmid_arc = rmid_pix * float(arcsec_per_pix)
    dr_arc = dr_pix * float(arcsec_per_pix)

    rmid_kpc = rmid_arc * float(scale_kpc_per_arcsec)
    dr_kpc = dr_arc * float(scale_kpc_per_arcsec)

    vel = np.asarray(velocity_profile["v"], dtype=float)
    vel_err = np.asarray(velocity_profile.get("v_err", np.full_like(vel, np.nan)), dtype=float)

    n = min(len(rmid_arc), len(dr_arc), len(flux_shell), len(lum_shell), len(ne_shell), len(vel), len(vel_err), len(npix_shell))

    rmid_arc = rmid_arc[:n]
    dr_arc = dr_arc[:n]
    rmid_kpc = rmid_kpc[:n]
    dr_kpc = dr_kpc[:n]
    flux_shell = flux_shell[:n]
    lum_shell = lum_shell[:n]
    ne_shell = np.asarray(ne_shell[:n], dtype=float)
    vel = vel[:n]
    vel_err = vel_err[:n]
    npix_shell = npix_shell[:n]

    mass = ionized_mass_from_luminosity(
        line_id=line_id,
        luminosity_erg_s=lum_shell,
        ne_cm3=ne_shell,
        z_over_zsun=z_over_zsun,
    )

    mdot = mass_outflow_rate_msun_per_yr(
        mass_msun=mass,
        velocity_kms=vel,
        delta_r_kpc=dr_kpc,
    )

    pdot = momentum_rate_dyne(mdot, vel)
    edot = kinetic_power_erg_s(mdot, vel)

    return {
        "r_arcsec": rmid_arc,
        "dr_arcsec": dr_arc,
        "r_kpc": rmid_kpc,
        "dr_kpc": dr_kpc,
        "v_kms": vel,
        "v_err_kms": vel_err,
        "flux_erg_s_cm2": flux_shell,
        "lum_erg_s": lum_shell,
        "ne_cm3": ne_shell,
        "mass_msun": mass,
        "mdot_msun_yr": mdot,
        "pdot_dyne": pdot,
        "edot_erg_s": edot,
        "npix_shell": npix_shell,
    }


def energetics_profile_to_table(profile_dict):
    return Table(profile_dict)


def save_energetics_table_fits(profile_dict, filename):
    cols = []

    def _add_col(name, key):
        if key in profile_dict:
            arr = np.asarray(profile_dict[key], dtype=np.float32)
            if arr.ndim == 1:
                cols.append(fits.Column(name=name, format="E", array=arr))

    _add_col("R_ARCSEC", "r_arcsec")
    _add_col("DR_ARCSEC", "dr_arcsec")
    _add_col("R_KPC", "r_kpc")
    _add_col("DR_KPC", "dr_kpc")

    _add_col("V_KMS", "v_kms")
    _add_col("VERR_KMS", "v_err_kms")

    _add_col("FLUX", "flux_erg_s_cm2")
    _add_col("LUMINOSITY", "lum_erg_s")
    _add_col("NE_CM3", "ne_cm3")
    _add_col("M_MSUN", "mass_msun")
    _add_col("MDOT", "mdot_msun_yr")
    _add_col("PDOT", "pdot_dyne")
    _add_col("EDOT", "edot_erg_s")
    _add_col("NPIX", "npix_shell")

    _add_col("MERR_MSUN", "mass_err_msun")
    _add_col("MDOT_ERR", "mdot_err_msun_yr")
    _add_col("PDOT_ERR", "pdot_err_dyne")
    _add_col("EDOT_ERR", "edot_err_erg_s")

    _add_col("MDOT_LO", "mdot_lo_msun_yr")
    _add_col("MDOT_HI", "mdot_hi_msun_yr")
    _add_col("MDOT_MID", "mdot_mid_msun_yr")

    hdu_primary = fits.PrimaryHDU()
    hdu_table = fits.BinTableHDU.from_columns(cols, name="ENERGETICS")

    if "density_mode" in profile_dict:
        hdu_table.header["DENSMODE"] = str(profile_dict["density_mode"])

    if "assumed_ne_values_cm3" in profile_dict:
        vals = np.asarray(profile_dict["assumed_ne_values_cm3"], dtype=float)
        hdu_table.header["NEGRID"] = ",".join(f"{v:g}" for v in vals)

    fits.HDUList([hdu_primary, hdu_table]).writeto(filename, overwrite=True)



def _disc_profile_from_physical_mode(
    *,
    fit_mode,
    best_param,
    best_param_err,
    n_shells,
    rin_pix,
    rout_pix,
    arcsec_per_pix,
    scale_kpc_per_arcsec,
    R_nsc_pc=None,
    a_plu_pc=None,
):
    r_arcsec, xerr_arcsec = _shell_midpoints_and_halfwidths_arcsec(
        rin_pix=rin_pix,
        rout_pix=rout_pix,
        n_shells=n_shells,
        arcsec_per_pix=arcsec_per_pix,
    )

    fit_mode = str(fit_mode)

    if fit_mode == "disk_kepler":
        v = vkep_astropy(r_arcsec, None, None, [best_param, scale_kpc_per_arcsec])

        if np.isfinite(best_param_err) and best_param > 0:
            v_hi = vkep_astropy(r_arcsec, None, None, [best_param + best_param_err, scale_kpc_per_arcsec])
            v_lo = vkep_astropy(r_arcsec, None, None, [max(best_param - best_param_err, 1e-12), scale_kpc_per_arcsec])
            v_err = 0.5 * np.abs(v_hi - v_lo)
        else:
            v_err = np.full_like(v, np.nan)

    elif fit_mode == "NSC":
        if R_nsc_pc is None:
            raise ValueError("R_nsc_pc is required for NSC mode.")
        v = vnsc_astropy(r_arcsec, None, None, [best_param, scale_kpc_per_arcsec, R_nsc_pc])

        if np.isfinite(best_param_err) and best_param > 0:
            v_hi = vnsc_astropy(r_arcsec, None, None, [best_param + best_param_err, scale_kpc_per_arcsec, R_nsc_pc])
            v_lo = vnsc_astropy(r_arcsec, None, None, [max(best_param - best_param_err, 1e-12), scale_kpc_per_arcsec, R_nsc_pc])
            v_err = 0.5 * np.abs(v_hi - v_lo)
        else:
            v_err = np.full_like(v, np.nan)

    elif fit_mode == "Plummer":
        if a_plu_pc is None:
            raise ValueError("a_plu_pc is required for Plummer mode.")
        v = vplummer_astropy(r_arcsec, None, None, [best_param, scale_kpc_per_arcsec, a_plu_pc])

        if np.isfinite(best_param_err) and best_param > 0:
            v_hi = vplummer_astropy(r_arcsec, None, None, [best_param + best_param_err, scale_kpc_per_arcsec, a_plu_pc])
            v_lo = vplummer_astropy(r_arcsec, None, None, [max(best_param - best_param_err, 1e-12), scale_kpc_per_arcsec, a_plu_pc])
            v_err = 0.5 * np.abs(v_hi - v_lo)
        else:
            v_err = np.full_like(v, np.nan)

    else:
        raise ValueError(f"Unsupported physical disc mode: {fit_mode}")

    return {
        "r_arcsec": r_arcsec,
        "xerr_arcsec": xerr_arcsec,
        "v": np.asarray(v, float),
        "v_err": np.asarray(v_err, float),
    }





def _is_rotation_curve_still_rising(disc_prof, outer_fraction=0.3):
    r = np.asarray(disc_prof["r_arcsec"], dtype=float)
    v = np.asarray(disc_prof["v"], dtype=float)
    good = np.isfinite(r) & np.isfinite(v)
    if np.sum(good) < 3:
        return None

    r = r[good]
    v = np.abs(v[good])

    order = np.argsort(r)
    r = r[order]
    v = v[order]

    n = len(r)
    n_outer = max(2, int(np.ceil(outer_fraction * n)))
    idx = np.arange(n - n_outer, n)

    rr = r[idx]
    vv = v[idx]

    if len(rr) < 2:
        return None

    slope = np.polyfit(rr, vv, 1)[0]
    return slope > 0







def _estimate_outer_vcirc(
    disc_prof,
    method="flat_plateau",
    outer_fraction=0.3,
    min_outer_points=2,
    flat_slope_frac=0.08,
    min_flat_points=3,
):
    """
    Estimate a representative outer circular velocity from the disc profile.

    Parameters
    ----------
    disc_prof : dict
        Must contain keys 'r_arcsec', 'v', 'v_err'.

    method : str
        Supported:
        - 'flat_plateau' : preferred; find approximately flat part of the curve
        - 'median_outer' : old behavior, median of outermost points
        - 'max_smooth'   : maximum of lightly smoothed profile

    outer_fraction : float
        Used only by fallback methods.

    min_outer_points : int
        Minimum number of points in outer methods.

    flat_slope_frac : float
        Threshold for "flatness", expressed as:
            |dv/dr| < flat_slope_frac * vmax / rmax
        Smaller = stricter flatness criterion.

    min_flat_points : int
        Minimum number of consecutive flat points required.

    Returns
    -------
    v_c_outer, v_c_outer_err, meta
    """
    r = np.asarray(disc_prof["r_arcsec"], dtype=float)
    v = np.asarray(disc_prof["v"], dtype=float)
    e = np.asarray(disc_prof["v_err"], dtype=float)

    good = np.isfinite(r) & np.isfinite(v)
    if np.sum(good) == 0:
        return np.nan, np.nan, {"method": method, "n_used": 0}

    r = r[good]
    v = np.abs(v[good])
    e = e[good] if e.shape == good.shape else np.full_like(v, np.nan, dtype=float)

    order = np.argsort(r)
    r = r[order]
    v = v[order]
    e = e[order]

    n = len(r)

    # --------------------------------------------------
    # Preferred method: find an approximately flat plateau
    # --------------------------------------------------
    if method == "flat_plateau":
        if n < 5:
            # too few points -> fallback
            return _estimate_outer_vcirc(
                disc_prof,
                method="median_outer",
                outer_fraction=outer_fraction,
                min_outer_points=min_outer_points,
            )

        # light smoothing
        kernel = np.array([0.25, 0.5, 0.25])
        v_pad = np.pad(v, (1, 1), mode="edge")
        v_smooth = np.convolve(v_pad, kernel, mode="valid")

        # local slope dv/dr
        dvdr = np.gradient(v_smooth, r)

        rmax = np.nanmax(r)

        # robust velocity scale: avoid being dominated by the noisy outer peak
        vmax_ref = np.nanpercentile(v_smooth, 70)

        if not np.isfinite(vmax_ref) or not np.isfinite(rmax) or rmax <= 0:
            return np.nan, np.nan, {"method": method, "n_used": 0}

        slope_thr = flat_slope_frac * vmax_ref / rmax

        # candidate flat points:
        #   small slope
        #   reasonably high velocity, but relative to a robust scale
        flat_mask = (np.abs(dvdr) <= slope_thr) & (v_smooth >= 0.80 * vmax_ref)



        logger.info(
            "flat_plateau candidates: %d / %d points (vmax_ref=%.2f)",
            int(np.sum(flat_mask)), len(flat_mask), vmax_ref
        )

        # search for consecutive flat segments
        idx = np.where(flat_mask)[0]

        if idx.size > 0:
            splits = np.where(np.diff(idx) > 1)[0] + 1
            groups = np.split(idx, splits)

            # keep only long enough groups
            groups = [g for g in groups if len(g) >= min_flat_points]

            if len(groups) > 0:
                # choose the longest flat group; if tie, prefer the one with lower scatter
                # and smaller radius (earlier plateau rather than noisy outer tail)
                gbest = sorted(
                    groups,
                    key=lambda g: (-len(g), np.nanstd(v[g]) if len(g) > 1 else np.inf, np.nanmedian(r[g]))
                )[0]

                v_outer = float(np.nanmedian(v[gbest]))

                if np.all(~np.isfinite(e[gbest])):
                    v_outer_err = np.nanstd(v[gbest], ddof=1) if len(gbest) > 1 else np.nan
                else:
                    finite_e = e[gbest][np.isfinite(e[gbest])]
                    scatter = np.nanstd(v[gbest], ddof=1) if len(gbest) > 1 else 0.0
                    if finite_e.size > 0:
                        v_outer_err = float(np.sqrt(np.nanmedian(finite_e**2) + scatter**2))
                    else:
                        v_outer_err = scatter if len(gbest) > 1 else np.nan

                return v_outer, v_outer_err, {
                    "method": "flat_plateau",
                    "n_used": int(len(gbest)),
                    "r_min_used_arcsec": float(r[gbest[0]]),
                    "r_max_used_arcsec": float(r[gbest[-1]]),
                    "slope_threshold": float(slope_thr),
                }


        logger.info(
            "flat_plateau found no valid plateau: vmax_ref=%.2f, rmax=%.2f, slope_thr=%.4f",
            vmax_ref, rmax, slope_thr
        )

        # fallback if no plateau found
        return _estimate_outer_vcirc(
            disc_prof,
            method="median_outer",
            outer_fraction=outer_fraction,
            min_outer_points=min_outer_points,
        )

    # --------------------------------------------------
    # Old method: median of outermost points
    # --------------------------------------------------
    elif method == "median_outer":
        n_outer = max(min_outer_points, int(np.ceil(outer_fraction * n)))
        n_outer = min(n_outer, n)
        idx = np.arange(n - n_outer, n)

        v_outer = np.nanmedian(v[idx])

        if np.all(~np.isfinite(e[idx])):
            v_outer_err = np.nanstd(v[idx], ddof=1) if n_outer > 1 else np.nan
        else:
            finite_e = e[idx][np.isfinite(e[idx])]
            scatter = np.nanstd(v[idx], ddof=1) if n_outer > 1 else 0.0
            if finite_e.size > 0:
                v_outer_err = np.sqrt(np.nanmedian(finite_e**2) + scatter**2)
            else:
                v_outer_err = scatter if n_outer > 1 else np.nan

        return v_outer, v_outer_err, {
            "method": "median_outer",
            "n_used": int(n_outer),
            "r_min_used_arcsec": float(r[idx[0]]),
            "r_max_used_arcsec": float(r[idx[-1]]),
        }

    # --------------------------------------------------
    # Max of smoothed curve
    # --------------------------------------------------
    elif method == "max_smooth":
        if n == 1:
            return float(v[0]), float(e[0]) if np.isfinite(e[0]) else np.nan, {
                "method": "max_smooth",
                "n_used": 1,
                "r_min_used_arcsec": float(r[0]),
                "r_max_used_arcsec": float(r[0]),
            }

        kernel = np.array([0.25, 0.5, 0.25])
        v_pad = np.pad(v, (1, 1), mode="edge")
        v_smooth = np.convolve(v_pad, kernel, mode="valid")
        i = int(np.nanargmax(v_smooth))

        v_outer = float(v_smooth[i])
        v_outer_err = float(e[i]) if np.isfinite(e[i]) else np.nan

        return v_outer, v_outer_err, {
            "method": "max_smooth",
            "n_used": 1,
            "r_min_used_arcsec": float(r[i]),
            "r_max_used_arcsec": float(r[i]),
        }

    else:
        raise ValueError(f"Unsupported method for outer vcirc estimation: {method}")






def _make_map_header_from_obs(obs, bunit=""):
    hdr = fits.Header()

    crpix = obs.cube.get("crpix", None)
    crval = obs.cube.get("crval", None)
    cdelt = obs.cube.get("cdelt", None)

    hdr["NAXIS"] = 2
    hdr["CTYPE1"] = "XOFFSET"
    hdr["CTYPE2"] = "YOFFSET"

    if crpix is not None and crval is not None and cdelt is not None:
        hdr["CRPIX1"] = float(crpix[2])
        hdr["CRPIX2"] = float(crpix[1])

        hdr["CRVAL1"] = float(crval[2])
        hdr["CRVAL2"] = float(crval[1])

        hdr["CDELT1"] = float(cdelt[2])
        hdr["CDELT2"] = float(cdelt[1])

    hdr["CUNIT1"] = "arcsec"
    hdr["CUNIT2"] = "arcsec"

    if bunit:
        hdr["BUNIT"] = bunit

    return hdr




def _make_cube_header_from_obs(obs):
    hdr = fits.Header()

    crpix = obs.cube.get("crpix", None)
    crval = obs.cube.get("crval", None)
    cdelt = obs.cube.get("cdelt", None)

    hdr["NAXIS"] = 3

    # FITS axis order is opposite to NumPy array order.
    # If data shape is (spec, y, x), then:
    #   FITS axis 1 -> x
    #   FITS axis 2 -> y
    #   FITS axis 3 -> spectral

    hdr["CTYPE1"] = "XOFFSET"
    hdr["CTYPE2"] = "YOFFSET"
    hdr["CTYPE3"] = "VELO-LSR"

    if crpix is not None and crval is not None and cdelt is not None:
        hdr["CRPIX1"] = float(crpix[2])
        hdr["CRPIX2"] = float(crpix[1])
        hdr["CRPIX3"] = float(crpix[0])

        hdr["CRVAL1"] = float(crval[2])
        hdr["CRVAL2"] = float(crval[1])
        hdr["CRVAL3"] = float(crval[0])

        hdr["CDELT1"] = float(cdelt[2])
        hdr["CDELT2"] = float(cdelt[1])
        hdr["CDELT3"] = float(cdelt[0])

    hdr["CUNIT1"] = "arcsec"
    hdr["CUNIT2"] = "arcsec"
    hdr["CUNIT3"] = "km/s"

    return hdr






def _save_escape_fraction_table_fits(profiles_dict, output_path: Path):
    hdus = [fits.PrimaryHDU()]

    for extname, prof in profiles_dict.items():
        if prof is None:
            continue

        cols = [
            fits.Column(name="R_ARCSEC", format="E", array=np.asarray(prof["r_arcsec"], dtype=np.float32)),
            fits.Column(name="XERR_ARCSEC", format="E", array=np.asarray(prof["xerr_arcsec"], dtype=np.float32)),
            fits.Column(name="RATIO", format="E", array=np.asarray(prof["ratio"], dtype=np.float32)),
            fits.Column(name="RATIO_ERR", format="E", array=np.asarray(prof["ratio_err"], dtype=np.float32)),
        ]

        if "v_esc" in prof:
            cols.append(
                fits.Column(name="V_ESC_KMS", format="E", array=np.asarray(prof["v_esc"], dtype=np.float32))
            )

        if "eta" in prof:
            cols.append(
                fits.Column(name="ETA", format="E", array=np.asarray(prof["eta"], dtype=np.float32))
            )

        if "ratio_loweta" in prof:
            cols.append(
                fits.Column(name="RATIO_ETA10", format="E", array=np.asarray(prof["ratio_loweta"], dtype=np.float32))
            )

        if "ratio_higheta" in prof:
            cols.append(
                fits.Column(name="RATIO_ETA100", format="E", array=np.asarray(prof["ratio_higheta"], dtype=np.float32))
            )

        hdus.append(fits.BinTableHDU.from_columns(cols, name=extname))

    fits.HDUList(hdus).writeto(output_path, overwrite=True)


def _save_model_cube_fits(model, obs, output_path: Path):
    hdr = _make_cube_header_from_obs(obs)
    data = np.asarray(model.cube["data"], dtype=np.float32)

    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(output_path, overwrite=True)





def _save_moment_maps_fits(obs, model, output_path: Path):
    flux_data = np.asarray(obs.maps["flux"], dtype=np.float32)
    vel_data = np.asarray(obs.maps["vel"], dtype=np.float32)
    sig_data = np.asarray(obs.maps["sig"], dtype=np.float32)

    flux_model = np.asarray(model.maps["flux"], dtype=np.float32)
    vel_model = np.asarray(model.maps["vel"], dtype=np.float32)
    sig_model = np.asarray(model.maps["sig"], dtype=np.float32)

    flux_resid = flux_data - flux_model
    vel_resid = vel_data - vel_model
    sig_resid = sig_data - sig_model

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(flux_data,  header=_make_map_header_from_obs(obs, bunit="flux"), name="FLUX_DATA"),
        fits.ImageHDU(vel_data,   header=_make_map_header_from_obs(obs, bunit="km/s"), name="VEL_DATA"),
        fits.ImageHDU(sig_data,   header=_make_map_header_from_obs(obs, bunit="km/s"), name="SIG_DATA"),

        fits.ImageHDU(flux_model, header=_make_map_header_from_obs(obs, bunit="flux"), name="FLUX_MODEL"),
        fits.ImageHDU(vel_model,  header=_make_map_header_from_obs(obs, bunit="km/s"), name="VEL_MODEL"),
        fits.ImageHDU(sig_model,  header=_make_map_header_from_obs(obs, bunit="km/s"), name="SIG_MODEL"),

        fits.ImageHDU(flux_resid, header=_make_map_header_from_obs(obs, bunit="flux"), name="FLUX_RESID"),
        fits.ImageHDU(vel_resid,  header=_make_map_header_from_obs(obs, bunit="km/s"), name="VEL_RESID"),
        fits.ImageHDU(sig_resid,  header=_make_map_header_from_obs(obs, bunit="km/s"), name="SIG_RESID"),
    ])

    hdul.writeto(output_path, overwrite=True)




