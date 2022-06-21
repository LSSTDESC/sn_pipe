import numpy as np
import glob
import matplotlib.pyplot as plt


def pixData(fDir, dbName):

    search_path = '{}/{}/*.npy'.format(fDir, dbName)

    fis = glob.glob(search_path)

    res = None
    for fi in fis:
        tt = np.load(fi, allow_pickle=True)
        if res is None:
            res = tt
        else:
            res = np.concatenate((res, tt))

    return res


fDir = '../ObsPixelized_fbs_2.0_16'
dbName = 'baseline_v2.0_10yrs'

data = pixData(fDir, dbName)

print(data)

idx = data['healpixID'] == 1449
sel = data[idx]

fig, ax = plt.subplots()

ax.plot(sel['fieldRA'], sel['fieldDec'], 'ko')

plt.show()
