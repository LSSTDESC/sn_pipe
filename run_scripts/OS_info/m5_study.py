from sn_tools.sn_obs import load_obs
import numpy as np
import pandas as pd
from sn_tools.sn_obs import season
import matplotlib.pyplot as plt


def filter_allocation(obs, Nvisits=800, outName='WL_req.csv'):

    ntot = len(obs)
    r = []
    for b in 'ugrizy':
        idx = obs['filter'] == b
        sel = obs[idx]
        b_alloc = len(sel)/ntot
        nv = np.round(b_alloc*Nvisits, 0)
        r.append((b, np.round(100.*b_alloc), nv))
        print(b, np.round(100.*b_alloc, 1), int(nv))

    df = pd.DataFrame(r, columns=['band', 'filter_allocation', 'Nvisits'])

    df.to_csv('WL_req.csv', index=False)


def get_m5(obs):

    # select DDF
    obs_DD = None
    print(np.unique(obs['note']))
    ido = np.core.defchararray.find(
        obs['note'].astype(str), 'DD')
    idob = np.ma.asarray(list(map(lambda x: bool(x+1), ido)))
    obs_DD = obs[idob]

    print('ff', np.unique(obs_DD['note']))

    # add seasons*
    print('test', obs_DD.dtype.names)
    obs_DD = season(obs_DD)

    fig, ax = plt.subplots()
    idx = obs_DD['note'] == 'DD:COSMOS'
    sel = obs_DD[idx]
    ax.plot(sel['season'], sel['fiveSigmaDepth'], 'ko')
    plt.show()

    # estimate m5 per DD and per season

    obs_df = pd.DataFrame.from_records(obs_DD)
    var = 'fiveSigmaDepth'
    tt = obs_df.groupby(['note', 'season', 'filter']).apply(
        lambda x: pd.DataFrame({var: [x[var].median()]})).reset_index()

    print(tt)


dbDir = '../DB_Files'
dbName = 'baseline_v3.0_10yrs'
dbExtens = 'db'

dbName = 'draft_connected_v2.99_10yrs'
dbExtens = 'npy'

observations = load_obs(dbDir, dbName, dbExtens)

# filter_allocation(observations)

get_m5(observations)
