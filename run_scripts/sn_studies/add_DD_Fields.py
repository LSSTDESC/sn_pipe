import numpy as np
import pandas as pd
from sn_tools.sn_obs import DDFields, season
from sn_tools.sn_clusters import ClusterObs
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
tab = np.load('{}/{}.npy'.format(dbDir, dbName), allow_pickle=True)

# get DDFields
DDF = DDFields()

nclusters = 6


clusters = ClusterObs(tab, nclusters, dbName, DDF).dataclus

# tab = pd.DataFrame(np.copy(clusters.dataclus))
# estimate seasons
tabseas = pd.DataFrame()
for field in clusters['fieldName'].unique():
    idx = clusters['fieldName'] == field
    sel = clusters[idx]
    seas = season(sel.to_records())
    print(field, np.unique(seas['season']))
    tabseas = pd.concat((tabseas, pd.DataFrame(np.copy(seas))))

"""
colors = ['k', 'r', 'g', 'b', 'y', 'm']
fig, ax = plt.subplots()
for io, name in enumerate(tabseas['fieldName'].unique()):
    ii = tabseas['fieldName'] == name
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
