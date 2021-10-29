import numpy as np
from optparse import OptionParser
import pandas as pd

parser = OptionParser()

parser.add_option("--dbName", type="str", default='baseline_nexp1_v1.7_10yrs.npy',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str",
                  default='../DB_Files', help="db dir [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

dbName = opts.dbName
dbDir = opts.dbDir

tab = np.load('{}/{}'.format(dbDir, dbName))
tab = pd.DataFrame(tab)

mjd_min = tab['mjd'].min()
night_max = tab['night'].max()
nights = tab['night'].unique().tolist()

mjds = np.arange(mjd_min, mjd_min+night_max, 1.)

df = pd.DataFrame(mjds, columns=['MJD'])
df['night'] = df['MJD']-df['MJD'].min()+1
df['night'] = df['night'].astype(int)

idx = df['night'].isin(nights)

print(len(df), len(df[idx]), len(df[~idx]), len(nights))

outName = dbName.replace('.npy', '_no_obs.npy')

np.save(outName, df[~idx].to_records(index=False))
