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
parser.add_option("--outName", type=str, default='Summary_DD_pointings.hdf5',
                  help="data location dir [%default]")
parser.add_option("--save_nightly", type=int, default=0,
                  help="to save nightly results[%default]")

opts, args = parser.parse_args()

dbList = opts.dbList
outName = opts.outName
save_nightly = opts.save_nightly

toprocess = pd.read_csv(dbList, comment='#')

print('toprocess', toprocess)

restot = pd.DataFrame()
for i, vv in toprocess.iterrows():
    restab = Stat_DD_night(vv['dbDir'], vv['dbName'], vv['dbExtens']).summary
    if save_nightly:
        thepath = 'Summary_night_{}.hdf5'.format(vv['dbName'])
        restab.to_pandas().to_hdf(thepath, key='summary_night')
        print(restab)
    res = Stat_DD_season(restab)
    # astropy.io.misc.hdf5.write_table_hdf5(
    #    restab, file_data, path=thepath, overwrite=True, serialize_meta=True)
    restot = pd.concat((restot, res))

restot.to_hdf(outName, key='summary')
