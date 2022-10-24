import pandas as pd
import numpy as np
from optparse import OptionParser
import operator as op
from sn_plotter_metrics import plt


def get_moon(data, lunar_phase=40.):
    """
    Function to select moon obs and grab corresponding obs duration

    Parameters
    ---------------
    data: pandas df
      data to process
    lunar_phase: float, opt
       lunar phase below which obs are considered to be moon obs (default: 40%)

    Returns
    ----------
    median moon obs duration over a season
    """

    # get moon obs
    idx = data['moonPhase'] <= lunar_phase
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
    """
    Function to estimate season length from a set of obs

    Parameters
    ---------------
    grp: pandas df group
      data to process

    Returns
    ----------
    pandas df with the following cols: moon_length, night_min, night_max


    """

    nmin = grp['night'].min()
    nmax = grp['night'].max()

    return pd.DataFrame({'moon_length': [nmax-nmin+1],
                         'night_min': nmin,
                         'night_max': nmax})


def select_data(data, config, ope):
    """
    Function to select data

    Parameters
    ---------------
    data: array
      data to process
    config: dict
      selection criteria
    ope: operator
      operator used for selection

    Returns
    ----------
    filtered array of data

    """
    idx = True
    for key, vv in config.items():
        idx &= ope(data[key], vv)

    return data[idx]


def plot(data, varx='night', vary='config', legx='night', legy='', figtit='', line=[], highlight={}, labelsize_x=-1, labelsize_y=-1):
    """
    Function to make a plot

    Parameters
    ----------------
    data: array
      data to plot
    varx: str, opt
      xaxis variable to plot (default: night)
    vary: str, opt
      yaxis variable to plot (default: config)
    legx: str, opt
      xaxis legend (default: night)
    legy: str, opt
      yaxis legend (default: '')
    figtit: str, opt
      figure title (default: '')
    line: list(float), opt
      line to draw [xmin, xmax] (default: [])
    hightlight: dict, opt
      data to highlight (default: {})


    """
    fig, ax = plt.subplots(figsize=(17, 10))
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

    if labelsize_x != -1:
        ax.tick_params(axis='x', labelsize=labelsize_x)
    if labelsize_y != -1:
        ax.tick_params(axis='y', labelsize=labelsize_y)

    ax.grid()


parser = OptionParser()

parser.add_option("--fName", type=str, default='Summary_night_ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs.hdf5',
                  help="file to process [%default]")
parser.add_option("--field", type=str, default='DD:COSMOS',
                  help="field to display [%default]")
parser.add_option("--season", type=int, default=1,
                  help="season to display [%default]")
parser.add_option("--lunar_phase", type=int, default=40,
                  help="lunar_phase sel cut [%default]")


opts, args = parser.parse_args()

fName = opts.fName
season = opts.season
field = opts.field
lunar_phase = opts.lunar_phase

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
hh['config'] = dict(zip('ugrizy', [0, 2, 9, 37, 52, 21]))
#hh['config'] = dict(zip('ugrizy', [0, 2, 9, 1, 1, 1]))
hh['op'] = op.eq
hh['color'] = 'r'
highlight[1] = hh
hhb = {}
hhb['config'] = dict(zip('u', [1]))
hhb['op'] = op.ge
hhb['color'] = 'b'
highlight[2] = hhb

moon_dur = get_moon(sel, lunar_phase=lunar_phase)

print('Moon', moon_dur)
#plot(sel, figtit=figtit, highlight=highlight, labelsize_y=12)
plot(sel, figtit=figtit, highlight=[], labelsize_y=12)
plt.show()
plot(sel, varx='night', vary='nvisits',
     legx='night', legy='Nvisits', figtit=figtit, line=[], highlight=highlight)
plt.show()
plot(sel, varx='moonPhase', vary='config',
     legx='lunar phase', legy='', figtit=figtit, line=[], highlight=highlight)
plt.show()
plot(sel, varx='night', vary='moonPhase',
     legx='night', legy='lunar phase', figtit=figtit, line=[], highlight=highlight)
plt.show()
for b in 'ugrizy':
    vva = 'deltaT_{}_mean'.format(b)
    vvb = 'deltaT_{}_rms'.format(b)
    sel[vva] *= 24.*3600.
    sel[vvb] *= 24.*3600.
    idx = sel[vva] > 0.
    figtitb = figtit
    figtitb += '\n {}-band'.format(b)
    plot(sel[idx], varx='night', vary=vva,
         legx='night', legy='Mean $\Delta T_{visit}$ [sec]', figtit=figtitb, line=[], highlight=highlight)
    plot(sel[idx], varx='night', vary=vvb,
         legx='night', legy='RMS $\Delta T_{visit}$ [sec]', figtit=figtitb, line=[], highlight=highlight)


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
