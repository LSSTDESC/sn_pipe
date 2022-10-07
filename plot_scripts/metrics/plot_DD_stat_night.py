import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser


def plot(data, varx='night', vary='config', legx='night', legy='', figtit=''):

    fig, ax = plt.subplots(figsize=(17, 8))
    fig.suptitle(figtit)

    ax.plot(data[varx], data[vary], 'ko')
    ax.set_xlabel(legx)
    ax.grid()


parser = OptionParser()

parser.add_option("--fName", type=str, default='Summary_night_ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs.hdf5',
                  help="file to process [%default]")
parser.add_option("--field", type=str, default='DD:COSMOS',
                  help="field to display [%default]")
parser.add_option("--season", type=int, default=1,
                  help="season to display [%default]")

opts, args = parser.parse_args()

fName = opts.fName
season = opts.season
field = opts.field

dbName = fName.split('.hdf5')[0].replace('Summary_night_', '')
df = pd.read_hdf(fName)

idx = df['field'] == field
idx &= df['season'] == season

sel = df[idx]
figtit = '{} - season {}'.format(field.split(':')[1], int(season))
figtit += '\n {}'.format(dbName)
sel = sel.sort_values(by=['night'])

figtitb = figtit
cad = np.mean(np.diff(sel['night']))
figtit += '\n mean cad: {} days'.format(np.round(cad, 1))
plot(sel, figtit=figtit)
dd = sel.groupby(['config']).size().to_frame('nnights').reset_index()

nn = 'N$_{nights}$'
figtitb += '\n {}={}'.format(nn, len(sel))
plot(dd, varx='nnights', legx='N$_{nights}$', figtit=figtitb)
io = dd['nnights'] <= 10
print(len(dd[io]), len(dd[io])/len(sel), dd[~io][['nnights', 'config']])

iu = sel['config'].isin(dd[~io]['config'].to_list())
iub = sel['config'].isin(['0u-2g-9r-37i-52z-21y'])

print('new cad', np.mean(np.diff(sel[iu]['night'])), np.mean(
    np.diff(sel[iub]['night'])))

plt.show()
