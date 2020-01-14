from sn_reshuffling.sn_reshuffling import StatSim
import numpy as np
from scipy import interpolate

def nVisits(grp,ref):

    for b in 'rizy':
        interp = interpolate.interp1d(ref['cadence'],ref['Nvisits_{}'.format(b)],bounds_error=False,fill_value=0.0)
        grp['Nvisits_new_{}'.format(b)] = int(np.round(interp(grp['cadence'])))


    return grp


dbDir='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.3'
dbName='euclid_ddf_v1.3_10yrs'
#dbName='descddf_illum10_v1.3_10yrs'
dbExtens='db'
nclusters = 6


# get data stat and median per night
datastat = StatSim(dbDir,dbName,dbExtens,nclusters)

# these are the stats
stat = datastat.data

# from this: grab the requested number of visits as a function of the redshift

visits_ref = np.load('Nvisits_cadence_Nvisits_median_m5_filter.npy')

print(visits_ref.dtype)
print(stat.columns)
for zlim in [0.75]:
    idx = np.abs(visits_ref['z']-zlim)<1.e-5
    sel_visits = visits_ref[idx]
    
    #print('yes',sel_visits)
    #print(stat['clusId', 'fieldName', 'season', 'season_length', 'cadence'].apply(lambda x: nVisits(x,sel_visits),axis=1))
    extr = stat[['clusId', 'fieldName', 'season', 'season_length', 'cadence']].copy()
    extr['Nvisits'] = 0
    for b in 'rizy':
        interp = interpolate.interp1d(sel_visits['cadence'],sel_visits['Nvisits_{}'.format(b)],bounds_error=False,fill_value=0.0)
        print('ahahahahah',extr['cadence'].values)
        print('rrr',np.round(interp(extr['cadence'].values)))
        nv = np.round(interp(extr['cadence'].values)).astype(int)
        extr['Nvisits_{}'.format(b)] = nv
        extr['Nvisits'] += nv

    print(extr)



"""
print(datastat.data)
print(datastat.mednight)
"""
