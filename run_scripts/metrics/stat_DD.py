import numpy as np
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_cadence_tools import Stat_DD_night, Stat_DD_season
import astropy
import h5py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = OptionParser()
parser.add_option("--dbList", type="str", default='List.csv',
                  help="db name [%default]")

opts, args = parser.parse_args()

dbList = opts.dbList

toprocess = pd.read_csv(dbList, comment='#')

print('toprocess', toprocess)
"""
outName = 'Summary_DD_night.hdf5'
file_data_night = h5py.File('{}'.format(outName), 'w')
"""

restot = pd.DataFrame()
for i, vv in toprocess.iterrows():
    restab = Stat_DD_night(vv['dbDir'], vv['dbName'], vv['dbExtens']).summary
    thepath = 'summary_{}'.format(i)
    print(restab)
    res = Stat_DD_season(restab)
    # astropy.io.misc.hdf5.write_table_hdf5(
    #    restab, file_data, path=thepath, overwrite=True, serialize_meta=True)
    restot = pd.concat((restot, res))

restot.to_hdf('Summary_DD.hdf5', key='summary')
