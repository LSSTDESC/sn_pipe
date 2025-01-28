#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:09:06 2025

@author: philippe.gris@clermont.in2p3.fr
"""
# from astropy.coordinates import SkyCoord
# from astropy.coordinates import Longitude, Latitude  # Angles
import astropy.units as u
from astropy.time import Time
from optparse import OptionParser
from sn_scheduler.scheduler import StarAltTime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sn_tools.sn_io import checkDir


def process_night(stars_alt, year, month, day, targets, plot_it=False):
    """


    Parameters
    ----------
    stars_alt : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    plot_it : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    targets_info : TYPE
        DESCRIPTION.

    """
    """
    Function to process targets

    Parameters
    ----------
    stars_alt : StarAltTime instance
        The class where the calculation is made.
    year : int
        year of observation.
    month : int
        month of observation.
    day : int
        day of observation.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    targets_info : pandas df
        Targets with obs. info.

    """

    # grab the targets
    stars_alt.target_location(targets=targets)

    # get stars alt
    stars_alt(year=year, month=month, day=day)

    # grab star infos
    alt_min = 25.
    alt_max = 86.5
    airmass_max = 2.5

    targets_info = stars_alt.target_info(star_alt_min=alt_min*u.deg,
                                         star_alt_max=alt_max*u.deg,
                                         star_airmass_max=airmass_max)

    """
    print(targets_info.columns)
    print(targets_info[['target', 'mjd', 'mjd_per_min_p1',
          'mjd_per_max_p1', 'obs_duration [h]']])
    """
    # plot result here
    if plot_it:
        stars_alt.plot(star_alt_min=alt_min*u.deg,
                       star_alt_max=alt_max*u.deg,
                       star_airmass_max=airmass_max)
        # stars_alt.plot_airmass()
        plt.show()

    return targets_info


def process_mjd(stars_alt, mjd, targets, plot_it=False):
    """
    Function to process a night (mjd)

    Parameters
    ----------
    stars_alt : StarAltTime class
        instance of the StarAltTime used for calculations.
    mjd : float
        Modified Julian Date of the night to process.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    rr : pandas df
        Result of the night processing.

    """

    tm = Time('{}'.format(mjd), format='mjd')

    print(tm.ymdhms, mjd)
    year = tm.ymdhms[0]
    month = tm.ymdhms[1]
    day = tm.ymdhms[2]

    rr = process_night(stars_alt, year, month, day, targets, plot_it=plot_it)

    return rr


def process(mjd_min, mjd_max, stars_alt, targets, plot_it=False, outDir=''):
    """
    Function to process data

    Parameters
    ----------
    mjd_min : float
        min Modified Julian Date.
    mjd_max : float
        max Modified Julian Date.
    stars_alt : StarAltTime class
        Instance of the StarAltTime class.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.
    outDir: str, optional.
        Output dir. The default is ''

    Returns
    -------
    None.

    """

    mjds = np.arange(mjd_min, mjd_max+1, 1)
    res = pd.DataFrame()
    for mjd in mjds:
        rr = process_mjd(stars_alt, mjd, targets, plot_it=plot_it)
        res = pd.concat((res, rr))

    outName = '{}/ddf_scheduler_{}_{}.hdf5'.format(
        outDir, int(mjd_min), int(mjd_max))

    res.to_hdf(outName, key='schedule')


parser = OptionParser(description='Script to estimate the schedule of a field')

parser.add_option('--fieldName', type=str,
                  default='COSMOS',
                  help='field name [%default]')
parser.add_option('--RA_deg', type=float, default=150.10833,
                  help="field RA [deg]  [%default]")
parser.add_option("--Dec_deg", type=float, default=2.233611,
                  help="field Dec [deg] [%default]")
parser.add_option("--target_list", type=str, default='input/scheduler/ddf.csv',
                  help="list of targets [%default]")
parser.add_option("--mjd_min", type=int, default=60980,
                  help="survey start [%default]")
parser.add_option("--num_years", type=int, default=10,
                  help="number of years [%default]")
parser.add_option("--year_length", type=int, default=365,
                  help="year length [days] [%default]")
parser.add_option("--outDir", type=str, default='../ddf_scheduler',
                  help="output dir for the results [%default]")

opts, args = parser.parse_args()

llist = opts.target_list
mjd_min = opts.mjd_min
num_years = opts.num_years
year_length = opts.year_length
outDir = opts.outDir

# create output dir (if necessary)
checkDir(outDir)

# load targets
targets = pd.read_csv(llist, comment='#')
"""
c = SkyCoord(fieldRA*u.deg, fieldDec*u.deg, frame='icrs')
fc = c.to_string('hmsdms', precision=3)
print(fc)
"""
# StarAltTime instance
stars_alt = StarAltTime()

# mjdds = np.linspace(mjd_min, mjd_max, num=9).astype(int)
# print(mjdds)

# mjjds = [mjd_min, mjd_min+1]

for i in range(num_years):

    mjdmin = mjd_min+i*year_length+i

    mjdmax = mjdmin+year_length
    print(mjdmin, mjdmax)
    process(mjdmin, mjdmax, stars_alt, targets, plot_it=False, outDir=outDir)

"""
for i in range(len(mjdds)-1):

    mj_min = mjdds[i]
    if i > 0:
        mj_min += 1
    mj_max = mjdds[i+1]
    print(mj_min, mj_max)
    process(mj_min, mj_max, stars_alt, targets, plot_it=False)
"""
