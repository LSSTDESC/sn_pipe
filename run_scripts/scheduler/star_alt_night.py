#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:20:52 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_scheduler.scheduler import StarAltTime
import astropy.units as u
from astropy.time import Time
from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd


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


parser = OptionParser(
    description='Script to estimate the schedule of a set of fields')


parser.add_option("--target_list", type=str, default='input/scheduler/ddf.csv',
                  help="list of targets [%default]")
parser.add_option("--year", type=int, default=2025,
                  help="year of observation [%default]")
parser.add_option("--month", type=int, default=11,
                  help="month of observation [%default]")
parser.add_option("--day", type=int, default=1,
                  help="day of observation [%default]")


opts, args = parser.parse_args()

llist = opts.target_list
year = opts.year
month = opts.month
day = opts.day


# load targets
targets = pd.read_csv(llist, comment='#')

# StarAltTime instance
stars_alt = StarAltTime()

rr = process_night(stars_alt, year, month, day, targets, plot_it=True)
