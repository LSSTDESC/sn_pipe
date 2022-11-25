import numpy as np
from optparse import OptionParser
from sn_plotter_metrics import plt

parser = OptionParser()

parser.add_option("--dbDir", type=str, default='../DB_Files',
                  help="location dir of the file to process [%default]")
parser.add_option("--dbName", type=str, default='ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs.npy',
                  help="file to process [%default]")
parser.add_option("--field", type=str, default='DD:COSMOS',
                  help="field to display [%default]")
parser.add_option("--night", type=int, default=472,
                  help="night to display [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
field = opts.field
night = opts.night

# load input file

data = np.load('{}/{}'.format(dbDir, dbName))

# select DDF and night
idx = data['note'] == field
if night >= 0:
    idx &= data['night'] == night
ddf = data[idx]
print(np.unique(ddf['note']), ddf.dtype)
# plot

fig, ax = plt.subplots(figsize=(17, 8), nrows=2)
fig.subplots_adjust(wspace=0, hspace=0)
figtit = '{} - night {}'.format(field.split(':')[1], int(night))
figtit += '\n {}'.format(dbName.split('.npy')[0])
fig.suptitle(figtit)

ax[0].plot(ddf['mjd'], ddf['band'], 'ro', markersize=8)

#ax.set_xlabel('MJD [days]')
ax[0].set_ylabel('band')
ax[0].grid()
ax[0].set_xticklabels([])
ddf.sort(order=['mjd'])
print(np.unique(ddf[['numExposures', 'exptime', 'visitTime']]))
ax[1].plot(ddf['mjd'], np.cumsum(ddf['numExposures']), 'k.')
ax[1].grid()
ax[1].set_xlabel('MJD [days]')
plt.show()
