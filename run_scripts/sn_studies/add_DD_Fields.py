import numpy as np
import pandas as pd
from sn_tools.sn_obs import DDFields, season
from sn_tools.sn_cadence_tools import Match_DD
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--dbName", type="str", default='flexddf_v1.4_10yrs_DD',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str",
                  default='../../../DB_Files', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='.',
                  help="output dir [%default]")
parser.add_option("--outputType", type='str', default='all',
                  help="output type: all or medians [%default]")


opts, args = parser.parse_args()

dbName = opts.dbName
dbDir = opts.dbDir
outDir = opts.outDir
outputType = opts.outputType

# store Data in pandas df
tab = pd.DataFrame(
    np.load('{}/{}.npy'.format(dbDir, dbName), allow_pickle=True))

# get DDFields
DDF = DDFields()

print('ddd', DDF)
tab.loc[:, 'pixRA'] = tab['fieldRA']
tab.loc[:, 'pixDec'] = tab['fieldDec']

# match DD to observations
tab = Match_DD(DDF, tab, radius=3.)

# estimate seasons
tabseas = pd.DataFrame()
for field in tab['fieldname'].unique():
    idx = tab['fieldname'] == field
    sel = tab[idx]
    seas = season(sel.to_records(index=False))
    print(field, np.unique(seas['season']))
    tabseas = pd.concat((tabseas, pd.DataFrame(np.copy(seas))))

"""
colors = ['k', 'r', 'g', 'b', 'y', 'm']
fig, ax = plt.subplots()
for io, name in enumerate(tabseas['fieldname'].unique()):
    ii = tabseas['fieldname'] == name
    sel = tabseas[ii]
    ax.plot(sel['fieldRA'], sel['fieldDec'], '{}.'.format(colors[io]))
plt.show()
"""

if outputType == 'all':
    # save without medians
    np.save('{}_with_fields.npy'.format(dbName),
            tabseas.to_records(index=False))

if outputType == 'median':
    # get median values
    groups = tabseas.groupby(
        ['fieldname', 'season', 'filter']).median().reset_index()

    print(groups)

    # save in npy file
    np.save('medValues_{}.npy'.format(dbName), groups.to_records(index=False))
