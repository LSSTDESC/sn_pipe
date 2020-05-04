import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))


def plot(sel, whatx='m5'):
    fig, ax = plt.subplots()
    for b in 'izy':
        idxb = sel['band'] == b
        selb = sel[idxb]
        ax.plot(selb[whatx], selb['SNR'], color=filtercolors[b],
                label='{}-band'.format(b), linestyle='None', marker='o')

    ax.legend()
    ax.grid()
    ax.set_xlabel(whatx)
    ax.set_ylabel('SNR')


def load(fname):
    return pd.DataFrame(np.load(fname, allow_pickle=True))


tab = load('reference_files/SNR_m5.npy')
meds = load('input/sn_studies/medValues_flexddf_v1.4_10yrs_DD.npy')

med_meds = meds.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()

print(med_meds)

# print(tab)
zref = 0.7
idx = np.abs(tab['z']-zref) < 1.e-5

idx &= tab['m5'] > 20.
idx &= tab['SNR'] < 50
sel = tab[idx]


sel['m5_single'] = sel.apply(
    lambda x: med_meds[med_meds['filter'] == x['band']]['fiveSigmaDepth'].values[0], axis=1)

print(sel.columns, sel[['m5', 'm5_single']])
sel.loc[:, 'Nvisits'] = 10**(0.8*(sel['m5']-sel['m5_single']))

print(sel)
plot(sel)
plot(sel, 'Nvisits')


plt.show()
