from sn_reshuffling.sn_reshuffling import StatSim


dbDir='/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.3'
dbName='euclid_ddf_v1.3_10yrs'
#dbName='descddf_illum10_v1.3_10yrs'
dbExtens='db'
nclusters = 6


datastat = StatSim(dbDir,dbName,dbExtens,nclusters)

print(datastat.data)
