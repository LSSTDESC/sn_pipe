import numpy as np
import pandas as pd
from sn_plotter_cosmology import plt


def load_field(dbDir, dbName, field):

    fullName = '{}/{}.npy'.format(dbDir, dbName)

    tt = np.load(fullName)

    return data_field(tt, field)


def data_field(tt, field):

    idx = tt['note'] == field

    return tt[idx]


def nvisits(data, season_min=1, season_max=10):

    idx = data['season'] >= season_min
    idx &= data['season'] <= season_max

    return np.sum(data[idx]['numExposures'])


dbDir = '../DB_Files'
df = pd.read_csv('fich_DDF.csv')


r = []
for i, row in df.iterrows():

    field_UD = load_field(dbDir, row['dbName'], row['field'])
    field_DD = load_field(dbDir, row['dbName'], 'DD:ECDFS')

    nvisits_UD = nvisits(field_UD, row['season_min'], row['season_max'])
    nvisits_DD = nvisits(field_DD)

    nseasons = row['season_max']-row['season_min']+1
    print(row['dbName'], nvisits_UD, nvisits_UD/nvisits_DD)
    r.append((row['dbName'], nvisits_UD/nvisits_DD, nseasons,
              row['marker'], row['mfc']))


dd = pd.DataFrame(
    r, columns=['dbName', 'ud_depth', 'nseasons', 'marker', 'mfc'])


fig, ax = plt.subplots(figsize=(14, 10))
fig.subplots_adjust(bottom=0.3)

ls = dict(zip([2, 3, 4], ['solid', 'dashed', 'dotted']))
nnseas = '$N_{seasons}$'
for nseas in dd['nseasons'].unique():
    idx = dd['nseasons'] == nseas
    sel = dd[idx]
    sel = sel.sort_values(by=['ud_depth'])
    ax.plot(sel['dbName'], sel['ud_depth'],
            linestyle=ls[nseas], color='k', lw=3,
            label='{}={}'.format(nnseas, nseas))

for dbName in dd['dbName'].unique():
    idx = dd['dbName'] == dbName
    selb = dd[idx]
    print('lll', dbName, selb['mfc'].values[0])
    ax.plot(selb['dbName'], selb['ud_depth'],
            marker=selb['marker'].values[0],
            color=selb['mfc'].values[0], ms=20,
            markerfacecolor=selb['mfc'].values[0])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode='anchor', fontsize=15)

ax.set_ylabel(r'$\frac{N_{visits}^{UD}}{N_{visits}^{DD}}$', fontsize=30)
ax.legend(bbox_to_anchor=(0.80, 1.1), ncol=3, frameon=False)

ax.grid()
plt.tight_layout()
plt.show()
