# import numpy as np
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_cadence_tools import Stat_DD_night, Stat_DD_season_night
from sn_tools.sn_cadence_tools import load_observations, get_fields
from sn_tools.sn_cadence_tools import stat_dd_season
from sn_tools.sn_utils import clean_level

# import astropy
# import h5py
from sn_tools.sn_io import checkDir
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = OptionParser()
parser.add_option("--dbList", type="str", default='List.csv',
                  help="db name [%default]")
parser.add_option("--outName", type=str, default='Summary_DD_pointings.hdf5',
                  help="data location dir [%default]")
parser.add_option("--save_nightly", type=int, default=0,
                  help="to save nightly results[%default]")
parser.add_option("--outDir", type=str, default='../summary_DD_pointings',
                  help="data outputdir [%default]")
parser.add_option("--lookuptable", type=str, default='input/simulation/lookup_ddf.csv',
                  help="lookup table for DDFs[%default]")

opts, args = parser.parse_args()

dbList = opts.dbList
outName = opts.outName
save_nightly = opts.save_nightly
outDir = opts.outDir
lookuptable = opts.lookuptable

checkDir(outDir)

toprocess = pd.read_csv(dbList, comment='#')

print('toprocess', toprocess)

restot = pd.DataFrame()
for i, vv in toprocess.iterrows():
    print('processing', vv['dbName'])
    obs = load_observations(vv['dbDir'], vv['dbName'], vv['dbExtens'])
    obs_dd = get_fields(obs, lookuptable)
    restab = Stat_DD_night(vv['dbName'], obs, obs_dd).summary
    if save_nightly:
        thepath = 'Summary_night_{}.hdf5'.format(vv['dbName'])
        restab.to_pandas().to_hdf(thepath, key='summary_night')
        print(restab)
    res = Stat_DD_season_night(restab)
    resb = stat_dd_season(obs_dd)
    resb = clean_level(resb)
    res = res.merge(resb, left_on=['field', 'season'], right_on=[
                    'field', 'season'], suffixes=['', ''])
    # astropy.io.misc.hdf5.write_table_hdf5(
    #    restab, file_data, path=thepath, overwrite=True, serialize_meta=True)
    # restot = pd.concat((restot, res))
    outDir_fi = '{}/{}'.format(outDir, vv['dbName'])
    checkDir(outDir_fi)
    res.to_hdf('{}/{}'.format(outDir_fi, outName), key='summary')


# restot.to_hdf(outName, key='summary')
