import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import operator as op


def get_moon(data):

    # get moon obs
    idx = data['u'] > 0
    selmoon = data[idx]

    # remove the season column
    selmoon = selmoon.drop(columns=['season'])

    # restimate seasons
    from sn_tools.sn_obs import season
    selmoon = season(selmoon.to_records(index=False),
                     season_gap=3, mjdCol='night')
    moondata = pd.DataFrame(np.copy(selmoon))

    moonseason = moondata.groupby(['season']).apply(lambda x: season_length(x))

    res = 0.
    if len(moonseason) > 0:
        res = moonseason['moon_length'].median()

    return res


def season_length(grp):

    nmin = grp['night'].min()
    nmax = grp['night'].max()

    return pd.DataFrame({'moon_length': [nmax-nmin+1],
                         'night_min': nmin,
                         'night_max': nmax})


def select_data(data, config, ope):

    idx = True
    for key, vv in config.items():
        print('hello', key, vv)
        idx &= ope(data[key], vv)

    return data[idx]


def plot(data, varx='night', vary='config', legx='night', legy='', figtit='', line=[], highlight={}):

    fig, ax = plt.subplots(figsize=(17, 8))
    fig.suptitle(figtit)

    ax.plot(data[varx], data[vary], 'ko', mfc='None')
    if line:
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], line, color='r')
    if highlight:
        for key, vals in highlight.items():
            seldata = select_data(data, vals['config'], vals['op'])
            ax.plot(seldata[varx], seldata[vary], '{}*'.format(vals['color']))
    ax.set_xlabel(legx)
    ax.set_ylabel(legy)
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
print('hello', sel.columns)
sel['nvisits'] = sel['u']+sel['g']+sel['r']+sel['i']+sel['z']+sel['y']

highlight = {}

hh = {}
#hh['config'] = dict(zip('ugrizy', [0, 2, 9, 37, 52, 21]))
hh['config'] = dict(zip('ugrizy', [0, 2, 9, 1, 1, 1]))
hh['op'] = op.eq
hh['color'] = 'r'
highlight[1] = hh
hhb = {}
hhb['config'] = dict(zip('u', [1]))
hhb['op'] = op.ge
hhb['color'] = 'b'
highlight[2] = hhb

moon_dur = get_moon(sel)

print('Moon', moon_dur)
plot(sel, figtit=figtit, highlight=highlight)
plot(sel, varx='night', vary='nvisits',
     legx='night', legy='Nvisits', figtit=figtit, line=[], highlight=highlight)
plot(sel, varx='moonPhase', vary='config',
     legx='lunar phase', legy='', figtit=figtit, line=[], highlight=highlight)
plot(sel, varx='night', vary='moonPhase',
     legx='night', legy='lunar phase', figtit=figtit, line=[40, 40], highlight=highlight)
dd = sel.groupby(['config', 'u', 'g', 'r', 'i', 'z', 'y']
                 ).size().to_frame('nnights').reset_index()
print('hhhh', dd)
nn = 'N$_{nights}$'
figtitb += '\n {}={}'.format(nn, len(sel))
plot(dd, varx='nnights', legx='N$_{nights}$',
     figtit=figtitb, highlight=highlight)
io = dd['nnights'] <= 10

if len(sel) > 0:
    print(len(dd[io]), len(dd[io])/len(sel), dd[~io][['nnights', 'config']])

    iu = sel['config'].isin(dd[~io]['config'].to_list())
    iub = sel['config'].isin(['0u-2g-9r-37i-52z-21y'])

    print('new cad', np.mean(np.diff(sel[iu]['night'])), np.mean(
        np.diff(sel[iub]['night'])))

plt.show()
