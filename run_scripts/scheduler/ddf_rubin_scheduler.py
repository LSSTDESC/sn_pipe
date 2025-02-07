import os

import numpy as np
import matplotlib.pylab as plt
from sn_rubin_scheduler.ddf_presched import generate_ddf_scheduled_obs_new
import pandas as pd
from optparse import OptionParser
from sn_tools.sn_io import checkDir


def process_ddf_grid():
    """
    Function to process the file ddf_grid.npz 
    (file in rubin_sim_data/scheduler/ddf_grid.npz)

    Returns
    -------
    None.

    """

    from rubin_scheduler.utils import SURVEY_START_MJD, ddf_locations
    from rubin_scheduler.data import get_data_dir

    mjd0 = SURVEY_START_MJD
    print(mjd0)

    # Info about each DDF
    data_file = os.path.join(get_data_dir(), "scheduler", "ddf_grid.npz")
    ddfs = ddf_locations()
    ddf_data = np.load(data_file)
    ddf_grid = ddf_data["ddf_grid"].copy()
    ddf_data.close()

    # Just to make plotting easier later,
    # let's crop off time before the survey starts
    indx = np.where(ddf_grid["mjd"] < mjd0)[0].max()
    ddf_grid = ddf_grid[indx:]

    print(ddfs)

    # Handy info pre-computed about each DDF
    print(ddf_grid.dtype)

    process_ddf_example('ECDFS', ddfs, ddf_grid, mjd0)


def process_ddf_example(name, ddfs, ddf_grid, mjd0):
    """
    Example on how to plot results from ddf_grid+optimization of the DDF survey

    Parameters
    ----------
    name : str
        field name.
    ddfs : array
        ddf location.
    ddf_grid : array
        grid of ddf observations.
    mjd0 : float
        start survey MJD.

    Returns
    -------
    None.

    """

    from sn_rubin_scheduler.ddf_presched import optimize_ddf_times
    # Run a script that sets up a desired total number of sequences vs time,
    # Then fits observations given depth constraints
    mjds, night_mjd, cumulative_desired, cumulative_sched = optimize_ddf_times(
        name,
        ddfs[name][0],
        ddf_grid,
        mjd_start=mjd0)

    # night_mjd is just the rough time, then mjd is the time optimally placed
    # within that night.
    figb, axb = plt.subplots()
    figb.suptitle(name)
    axb.plot(night_mjd - mjd0, cumulative_desired,
             label='Desired', linewidth=7)
    axb.plot(night_mjd - mjd0, cumulative_sched,
             label='Best Fit', linewidth=3, color="r")
    axb.legend()
    axb.set_xlim([0, 365*4])
    axb.set_ylim([0, 600])
    axb.set_xlabel("night")
    axb.set_ylabel("Cumulative number of DDF Sequences")

    fig, ax = plt.subplots()
    fig.suptitle(name)
    ax.plot(ddf_grid["mjd"] - mjd0, ddf_grid["%s_m5_g" % name])
    m5_interp = np.interp(mjds, ddf_grid["mjd"], ddf_grid["%s_m5_g" % name])

    ax.plot(np.array(mjds)-mjd0, m5_interp, 'ro')

    ax.set_xlim([0, 730])
    ax.set_xlabel("night")
    ax.set_ylabel("g 5-sigma depth (mags)")


def ddf_config(sequence_time=60.0,
               season_unobs_frac=0.2,
               low_season_frac=0,
               low_season_rate=0.3,
               g_depth_limit=23.5):
    """
    Function to define the ddf configuration for the rubin scheduler

    Parameters
    ----------
    sequence_time : float, optional
        Expected time for each DDF sequence, used to avoid hitting the
        sun_limit (running DDF visits into twilight). In minutes. 
        The default is 60.0.
    season_unobs_frac : float, optional
        Defines the end of the range of the prescheduled observing season.
        season runs from 0 (sun's apparent position is at the RA of the DDF)
        to 1 (sun returns to an apparent position in the RA of the DDF).
       The scheduled season runs from:
       season_unobs_frac < season < (1-season_unobs_fract)
       The default is 0.2.
    low_season_frac : float, optional
        Defines the end of the range of the "low cadence" prescheduled
        observing season.
        The "standard cadence" season runs from:
        low_season_frac < season < (1 - low_season_frac)
        For an 'accordian' style DDF with fewer observations near
        the ends of the season, set this to a value larger than
        `season_unobs_frac`. Values smaller than `season_unobs_frac`
        will result in DDFs with a constant rate throughout the season.
        The default is 0.
    low_season_rate : float, optional
        Defines the rate to use within the low cadence portion
        of the season. During the standard season, the 'rate' is 1.
        This is used in `ddf_slopes` to define the desired number of
        cumulative observations for each DDF over time.
        The default is 0.3.
    g_depth_limit : float, optional
        The minimum g band five sigma depth limit allowed when prescheduling
        DDF visits. This is a depth limit in g band (mags).
        The depth is calculated using skybrightness from skybrightness_pre,
        a nominal FWHM_500 seeing at zenith of 0.7" (resulting in airmass
        dependent seeing) and exposure time.
        Default 23.5. Set to None for no limit.
        The default is 23.5.

    Returns
    -------
    ddf_kwargs : dict
        DDF parameters.

    """

    ddf_kwargs = {}

    ddf_kwargs["ELAISS1"] = {
        "season_seq": 80,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "g_depth_limit": g_depth_limit,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["XMM_LSS"] = {
        "season_seq": 110,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "g_depth_limit": g_depth_limit,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["ECDFS"] = {
        "season_seq": 80,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "g_depth_limit": g_depth_limit,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["COSMOS"] = {
        "season_seq": 110,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "g_depth_limit": g_depth_limit,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["EDFS_a"] = {
        "season_seq": 40,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "g_depth_limit": g_depth_limit,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    return ddf_kwargs


parser = OptionParser(
    description='Script to produce DDF observations for the LSST scheduler')

parser.add_option('--inputDir', type=str,
                  default='../observations_sn_scheduler',
                  help='Location dir of input files [%default]')
parser.add_option('--outputDir', type=str,
                  default='../desc_ddf_deep_rolling',
                  help='output dir of the produced files [%default]')
parser.add_option('--ddf_survey', type=str,
                  default='ddf_desc_0.70_sn',
                  help='name of the survey to produce [%default]')

opts, args = parser.parse_args()

inputDir = opts.inputDir
outputDir = opts.outputDir
ddf_survey = opts.ddf_survey


"""
process_ddf_grid()

plt.show()
"""

# check if output dir exist
checkDir(outputDir)

# grab ddf configuration for rubin survey
ddf_kwargs = ddf_config()


# grab ddf scenario
fName = '{}/{}.hdf5'.format(inputDir, ddf_survey)
ddf_scenario = pd.read_hdf(fName)

# generate observations

observations = generate_ddf_scheduled_obs_new(dist_tol=1,
                                              ddf_kwargs=ddf_kwargs,
                                              ddf_scenario=ddf_scenario,
                                              include_moon_phase=False)
# save the file
outName = '{}/{}.npy'.format(outputDir, ddf_survey)
np.save(outName, observations)
