import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

def get_all_files(start_hour=11, end_hour=3):
    """
    start_hour - in UTC
    end_hour - in UTC"""
    if end_hour < start_hour:
        hour_range = np.union1d(np.arange(start_hour, 23 + 1),
                                np.arange(0, end_hour + 1))
    else:
        hour_range = np.arange(start_hour, end_hour)
    files = []
    for hour in hour_range:
        temp = glob(
            ('/a2/uaren/goes_images/april/goes15.2014.*.{hour:02d}*.BAND_01.nc'
                          .format(hour=hour)))
        files = files + temp
        temp = glob(
            ('/a2/uaren/goes_images/may/goes15.2014.*.{hour:02d}*.BAND_01.nc'
                          .format(hour=hour)))
        files = files + temp
        temp = glob(
            ('/a2/uaren/goes_images/june/goes15.2014.*.{hour:02d}*.BAND_01.nc'
                          .format(hour=hour)))
        files = files + temp
    files.sort(key = lambda x: x.sort('.2014.')[1])
    return files


def sphere_to_lcc(self, lats, lons, R=6370, truelat0=31.7, truelat1=31.7,
                  ref_lat=31.68858, stand_lon=-113.7):
    """
    Taken from Tony Lorenzo's repository at:
    https://github.com/alorenzo175/
         satellite_irradiance_optimal_interpolation.git.
    Convert from spherical lats/lons like what comes out of WRF to the WRF
    Lambert Conformal x/y coordinates. Defaults are what
    are generally used for the AZ domain
    """
    phis = np.radians(lats)
    lambdas = np.radians(lons)
    phi0 = np.radians(ref_lat)
    phi1 = np.radians(truelat0)
    phi2 = np.radians(truelat1)
    lambda0 = np.radians(stand_lon)

    if truelat0 == truelat1:
        n = np.sin(phi0)
    else:
        n = (np.log(np.cos(phi1) / np.cos(phi2)) /
             np.log(np.tan(np.pi / 4 + phi2 / 2) /
                    np.tan(np.pi / 4 + phi1 / 2)
                    ))
    F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
    rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
    rho = F / np.power(np.tan(np.pi / 4 + phis / 2), n)
    x = R * rho * np.sin(n * (lambdas - lambda0))
    y = R * (rho0 - rho * np.cos(n * (lambdas - lambda0)))

    return x, y

def lcc_to_sphere(self, x, y, R=6370, truelat0=31.7, truelat1=31.7,
                  ref_lat=31.68858, stand_lon=-113.7):
    """
    Taken from Tony Lorenzo's repository at:
    https://github.com/alorenzo175/
         satellite_irradiance_optimal_interpolation.git.
    Convert from spherical lats/lons like what comes out of WRF to the WRF
    Lambert Conformal x/y coordinates. Defaults are what
    are generally used for the AZ domain
    """
    phi0 = np.radians(ref_lat)
    phi1 = np.radians(truelat0)
    phi2 = np.radians(truelat1)
    lambda0 = np.radians(stand_lon)

    if truelat0 == truelat1:
        n = np.sin(phi0)
    else:
        n = (np.log(np.cos(phi1) / np.cos(phi2)) /
             np.log(np.tan(np.pi / 4 + phi2 / 2) /
                    np.tan(np.pi / 4 + phi1 / 2)
                    ))
    F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
    rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
    x = x / R
    y = y /R
    rho = np.sqrt(x**2 + (y - rho0)**2)
    phis = 2 * (np.arctan2(F**(1.0 / n), rho**(1.0 / n)) - np.pi / 4)
    lambdas = np.arcsin(x / rho) / n + lambda0

    return np.degrees(phis), np.degrees(lambdas)
