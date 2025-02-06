#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:09:06 2025

@author: philippe.gris@clermont.in2p3.fr
"""
# from astropy.coordinates import SkyCoord
# from astropy.coordinates import Longitude, Latitude  # Angles

from optparse import OptionParser
from sn_scheduler.scheduler import StarAltTime
import pandas as pd
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import multiproc
from sn_scheduler.scheduler import process_multiproc
import warnings
warnings.filterwarnings("ignore")


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
parser.add_option("--outDir", type=str, default='../sn_ddf_scheduler',
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

mjds = []
for i in range(num_years):

    mjdmin = mjd_min+i*year_length+i

    mjdmax = mjdmin+year_length

    mjds.append((mjdmin, mjdmax))


"""

print((mjdmax-mjd_min)/year_length)


mjdds = np.linspace(mjd_min, mjd_min+11*year_length, num=9).astype(int)
print(mjdds)
print(test)
"""

params = {}
params['star_alt'] = stars_alt
params['targets'] = targets
params['outDir'] = outDir

multiproc(mjds, params, process_multiproc, nproc=8)

"""
for i in range(len(mjdds)-1):

    mj_min = mjdds[i]
    if i > 0:
        mj_min += 1
    mj_max = mjdds[i+1]
    print(mj_min, mj_max)
    process(mj_min, mj_max, stars_alt, targets, plot_it=False)
"""
