from sn_reshuffling.datastat import StatSim
from sn_reshuffling.obsresh import ObsReshuffled

import numpy as np
import os


dbDir='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.3'
dbName='euclid_ddf_v1.3_10yrs'
#dbName='descddf_illum10_v1.3_10yrs'
dbExtens='db'
nclusters = 6


# get data stat and median per night
datastat = StatSim(dbDir,dbName,dbExtens,nclusters)

# these are the stats
stat = datastat.data

print(stat)


# from this: grab the requested number of visits as a function of the redshift

visits_ref = np.load('Nvisits_cadence_Nvisits_median_m5_filter.npy')

print(visits_ref.dtype)
print(stat.columns)
for zlim in np.arange(0.5,0.85,0.05):
    ObsReshuffled(datastat.mednight, datastat.data,visits_ref,zlim,dbName)

"""
tab = np.load('euclid_ddf_v1.3_10yrs_0.75.npy')
print(tab.dtype)

fName ='euclid_ddf_v1.3_10yrs_0.75'
datastatb = StatSim(os.getcwd(),fName,'npy',nclusters) 

print(datastatb.data)
"""

