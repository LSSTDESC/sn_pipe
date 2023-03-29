from sn_tools.sn_obs import load_obs


dbDir = '../DB_Files'
dbName = 'baseline_v3.0_10yrs'
dbExtens = 'db'

dbName = 'draft_connected_v2.99_10yrs'
dbExtens = 'npy'

observations = load_obs(dbDir, dbName, dbExtens)

print(observations)
