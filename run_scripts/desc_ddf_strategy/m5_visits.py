import numpy as np
import pandas as pd


def load_DDF(dbDir, dbName, DDList=['COSMOS', 'ECDFS',
                                    'EDFS_a', 'EDFS_b', 'ELAISS1', 'XMM_LSS']):

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


def get_median_m5_field(tab):

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


def get_median_m5(tab):

    r = []
    for b in 'ugrizy':
        idxb = tab['band'] == b
        selb = tab[idxb]
        r.append((b, np.median(selb['fiveSigmaDepth'])))

    msingle = np.rec.fromrecords(r, names=['band', 'm5_med_single'])

    return msingle


def get_Nvisits(msingle, df_pz):

    msingle = msingle.merge(df_pz, left_on=['band'], right_on=['band'])
    """
    diffa = msingle['m5_y1']-msingle['m5_med_single']
    diffb = msingle['m5_y2_y10']-msingle['m5_med_single']
    diffc = msingle['m5_y2_y10_m']-msingle['m5_med_single']
    diffd = msingle['m5_y2_y10_p']-msingle['m5_med_single']

    msingle['Nvisits_y1'] = 10**(0.8 * diffa)
    msingle['Nvisits_y2_y10'] = 10**(0.8 * diffb)
    """
    llv = []

    for vv in ['y1', 'y2_y10', 'y2_y10_m', 'y2_y10_p']:
        diff = msingle['m5_{}'.format(vv)]-msingle['m5_med_single']
        Nv = 'Nvisits_{}'.format(vv)
        msingle[Nv] = 10**(0.8 * diff)
        llv.append(Nv)
    if 'field' in msingle.columns:
        summary = msingle.groupby(['field'])[llv].sum().reset_index()
    else:
        summary = msingle[llv].sum()

    return msingle, summary


sel = load_DDF('../DB_Files', 'draft_connected_v2.99_10yrs.npy')
print(np.unique(sel['note']))

msingle = get_median_m5(sel)
msingle = pd.DataFrame.from_records(msingle)

bands = 'ugrizy'
df_pz = pd.DataFrame([*bands], columns=['band'])
df_pz['m5_y1'] = [26.7, 27.0, 26.2, 25.8, 25.6, 24.7]
ll = [27.8, 28.1, 27.8, 27.6, 27.2, 26.5]
df_pz['m5_y2_y10'] = ll
delta_mag = 0.05
ll = list(map(lambda x: x - delta_mag, ll))
df_pz['m5_y2_y10_m'] = ll

ll = list(map(lambda x: x + 2*delta_mag, ll))
df_pz['m5_y2_y10_p'] = ll

msingle_calc, summary = get_Nvisits(msingle, df_pz)

print(summary)


ntot = msingle_calc['Nvisits_y2_y10'].sum()
r = []
Nvisits = 1480*9
Nvisits = 1550*9
# Nvisits = 14852
for b in 'ugrizy':
    idx = msingle_calc['band'] == b
    frac = msingle_calc[idx]['Nvisits_y2_y10'].values/ntot
    r.append((b, frac[0]*Nvisits))

df = pd.DataFrame(r, columns=['band', 'Nvisits'])
print(df)
df = df.merge(msingle, left_on=['band'], right_on=['band'])
df = df.merge(df_pz, left_on=['band'], right_on=['band'])
df['m5'] = df['m5_med_single']+1.25*np.log10(df['Nvisits'])
df['delta_m5'] = df['m5']-df['m5_y2_y10']

print(df[['band', 'm5', 'delta_m5']])
