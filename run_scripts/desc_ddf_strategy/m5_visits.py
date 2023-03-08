import numpy as np
import pandas as pd


def load_DDF(dbDir, dbName, DDList=['COSMOS', 'ECDFS', 'EDFS_a', 'EDFS_b', 'ELAISS1', 'XMM_LSS']):

    fullPath = '{}/{}'.format(dbDir, dbName)
    tt = np.load(fullPath)

    print(np.unique(tt['note']))
    data = None
    for field in DDList:
        idx = tt['note'] == 'DD:{}'.format(field)
        if data is None:
            data = tt[idx]
        else:
            data = np.concatenate((data, tt[idx]))

    return data


def get_median_m5(tab):

    r = []
    for field in np.unique(tab['note']):
        idxa = tab['note'] == field
        sela = tab[idxa]
        for b in 'ugrizy':
            idxb = sela['band'] == b
            selb = sela[idxb]
            print(b, np.median(selb['fiveSigmaDepth']))
            r.append(
                (b, np.median(selb['fiveSigmaDepth']), field.split(':')[-1]))

    msingle = np.rec.fromrecords(r, names=['band', 'm5_med_single', 'field'])

    return msingle


def get_Nvisits(msingle, df_pz):

    msingle = msingle.merge(df_pz, left_on=['band'], right_on=['band'])

    msingle['Nvisits_y1'] = 10**(0.8 *
                                 (msingle['m5_y1']-msingle['m5_med_single']))
    msingle['Nvisits_y2_y10'] = 10**(0.8 *
                                     (msingle['m5_y2_y10']-msingle['m5_med_single']))
    print(msingle)
    summary = msingle.groupby(['field'])['Nvisits_y1',
                                         'Nvisits_y2_y10'].sum().reset_index()

    return msingle, summary


sel = load_DDF('../DB_Files', 'draft_connected_v2.99_10yrs.npy')
print(np.unique(sel['note']))

msingle = get_median_m5(sel)
msingle = pd.DataFrame.from_records(msingle)

bands = 'ugrizy'
df_pz = pd.DataFrame([*bands], columns=['band'])
df_pz['m5_y1'] = [26.7, 27.0, 26.2, 25.8, 25.6, 24.7]
df_pz['m5_y2_y10'] = [27.8, 28.1, 27.8, 27.6, 27.2, 26.5]


msingle, summary = get_Nvisits(msingle, df_pz)

print(summary)
