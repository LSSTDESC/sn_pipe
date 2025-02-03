import os

import numpy as np
import matplotlib.pylab as plt
from sn_rubin_scheduler.ddf_presched import ddf_slopes, match_cumulative, optimize_ddf_times, generate_ddf_scheduled_obs

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.utils import ScheduledObservationArray
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import SURVEY_START_MJD, calc_season, ddf_locations


def process_ddf_example(name, ddfs, ddf_grid):

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
               low_season_rate=0.3):

    ddf_kwargs = {}

    ddf_kwargs["ELAISS1"] = {
        "season_seq": 30,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["XMM_LSS"] = {
        "season_seq": 30,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["ECDFS"] = {
        "season_seq": 30,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["COSMOS"] = {
        "season_seq": 30,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    ddf_kwargs["EDFS_a"] = {
        "season_seq": 30,
        "boost_early_factor": None,
        "boost_factor_third": 0,
        "season_unobs_frac": season_unobs_frac,
        "sequence_time": sequence_time,
        "low_season_frac": low_season_frac,
        "low_season_rate": low_season_rate,
    }

    return ddf_kwargs


"""
mjd0 = SURVEY_START_MJD
print(mjd0)


# Info about each DDF
data_file = os.path.join(get_data_dir(), "scheduler", "ddf_grid.npz")
ddfs = ddf_locations()
ddf_data = np.load(data_file)
ddf_grid = ddf_data["ddf_grid"].copy()
ddf_data.close()

# Just to make plotting easier later, let's crop off time before the survey starts
indx = np.where(ddf_grid["mjd"] < mjd0)[0].max()
ddf_grid = ddf_grid[indx:]

print(ddfs)

# Handy info pre-computed about each DDF
print(ddf_grid.dtype)

process_ddf_example('ECDFS', ddfs, ddf_grid)
"""

# grab ddf configuration
ddf_kwargs = ddf_config()

observations = generate_ddf_scheduled_obs(dist_tol=1, ddf_kwargs=ddf_kwargs)
print(observations)
np.save('observations_scheduler.npy', observations)

plt.show()
