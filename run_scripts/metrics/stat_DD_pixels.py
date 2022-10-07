import pandas as pd
import numpy as np
import os
import glob
from sn_tools.sn_cadence_tools import stat_DD_night_pixel, stat_DD_season_pixel
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--dbDir", type=str, default='../ObsPixelized_128',
                  help="data location dir [%default]")
parser.add_option("--dbList", type=str, default='List_ObsPixels.csv',
                  help="List of db to process [%default]")
parser.add_option("--outName", type=str, default='Summary_DD_pixel.hdf5',
                  help="data location dir [%default]")

opts, args = parser.parse_args()

obsPixelDir = opts.dbDir
dbList = opts.dbList

dfdb = pd.read_csv(dbList, comment='#')

restot = pd.DataFrame()

for i, row in dfdb.iterrows():
    # processing obspixelfiles
    resdf = stat_DD_night_pixel(obsPixelDir, row['dbName'])
    restat = stat_DD_season_pixel(resdf)
    restot = pd.concat((restot, restat))

restot.to_hdf(opts.outName, key='summary')
