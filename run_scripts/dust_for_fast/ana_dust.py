import h5py
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

dirSimu = 'Output_Simu'

x1 = -2.0
color = 0.2

x1 = np.round(x1, 2)
color = np.round(color, 2)

prodid = 'sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_ebvofMW_0.0'.format(x1, color)
simuFile = '{}/Simu_{}.hdf5'.format(dirSimu, prodid)
LCFile = '{}/LC_{}.hdf5'.format(dirSimu, prodid)

proddust = []
pp = '_'.join(prodid.split('_')[:-2])

for ddv in np.arange(0.0, 0.06, 0.005):
    ddv = np.round(ddv, 2)
    if ddv < 0.06:
        proddust.append('{}_ebvofMW_{}'.format(pp, np.round(ddv, 2)))
"""
proddust = ['sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_1_0.01',
            'sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_1_0.02',
            'sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_1_0.03',
            'sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_1_0.04',
            'sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_1_0.05']
"""
LCdust = {}

for pp in proddust:
    LCdust[pp] = '{}/LC_{}.hdf5'.format(dirSimu, pp)


# getting the simu file
f = h5py.File(simuFile, 'r')
print(f.keys())
# reading the simu file
for i, key in enumerate(f.keys()):
    simu = Table.read(simuFile, path=key)

print(simu)

ebvs = []
lcall = {}
lcdust_all = {}
tocorr = ['flux', 'mag', 'flux_e_sec', 'snr_m5', 'magerr', 'fluxerr']

phases = np.around(np.arange(-20., 55., 0.1), 2)
tabtot = Table()
for tt in simu:
    lcref = Table.read(LCFile, path='lc_{}'.format(tt['index_hdf5']))
    print(lcref.columns)
    # print(lcref.meta['dust'], lcref.meta['ebvofMW'], lcref['flux_e_sec'])
    lctot = Table(lcref)
    for kk, vv in LCdust.items():
        lcb = Table.read(vv, path='lc_{}'.format(tt['index_hdf5']))

        for band in 'grizy':
            idxref = lcref['band'] == 'LSST::'+band
            selref = lcref[idxref]
            if len(selref) > 0:
                idx = lcb['band'] == 'LSST::'+band
                selb = lcb[idx]
                interpo = interp1d(selb['phase'], selb['flux_e_sec'] /
                                   selref['flux_e_sec'], bounds_error=False, fill_value=0.)
                """
                interpoa = interp1d(
                    selb['phase'], selb['flux_e_sec'], bounds_error=False, fill_value=0.)
                interporef = interp1d(
                    selb['phase'], selref['flux_e_sec'], bounds_error=False, fill_value=0.)
                """
                tab = Table()
                tab['phase'] = phases
                ratios = interpo(phases)
                ratios[np.isnan(ratios)] = 0.
                tab['ratio'] = ratios
                """
                tab['ref'] = interporef(phases)
                tab['other'] = interpoa(phases)
                """
                tab['band'] = band
                tab['z'] = np.round(lcb.meta['z'], 2)
                tab['ebvofMW'] = np.round(lcb.meta['ebvofMW'], 2)
                tabtot = vstack([tabtot, tab])

idd = tabtot['band'] == 'z'
idd &= tabtot['z'] == 0.09

print(tabtot[idd])

tabtot.write('Dust_{}_{}.hdf5'.format(x1, color), 'dust', compression=True)
print(tabtot)

print(test)


for tt in simu:
    lcref = Table.read(LCFile, path='lc_{}'.format(tt['index_hdf5']))
    print(lcref.columns)
    # print(lcref.meta['dust'], lcref.meta['ebvofMW'], lcref['flux_e_sec'])
    lctot = Table(lcref)
    for kk, vv in LCdust.items():
        if kk not in lcdust_all.keys():
            lcdust_all[kk] = {}
        lcb = Table.read(vv, path='lc_{}'.format(tt['index_hdf5']))
        zz = np.round(lcb.meta['z'], 2)
        for valc in tocorr:
            lcb['{}_ref'.format(valc)] = lctot[valc]
        lcb['ebvofMW'] = lcb.meta['ebvofMW']
        lcdust_all[kk][zz] = lcb

    lcall[np.round(lcref.meta['z'], 2)] = lctot
    # break
    # break

what = 'mag'
for b in 'grizy':
    fig, ax = plt.subplots()
    fig.suptitle('{} band'.format(b))
    for key, vals in lcdust_all.items():
        for keyb, valb in vals.items():
            idx = valb['band'] == 'LSST::'+b
            sell = valb[idx]

            ax.plot(sell['ebvofMW'], sell['{}_ref'.format(what)] /
                    sell[what], marker='o', mfc='None', color='k', lineStyle='None')
            ax.plot(sell['ebvofMW'], sell['flux_ref'] /
                    sell['flux'], marker='*', color='r', lineStyle='None')

plt.show()
print(lcall.keys())
for b in 'grizy':
    fig, ax = plt.subplots()
    fig.suptitle('{} band'.format(b))
    for zv in np.arange(0.1, 1., 0.1):
        z = np.round(zv, 2)
        lctot = lcall[z]
        idx = lctot['band'] == 'LSST::'+b
        sel = lctot[idx]
        for ebv in ebvs:
            fdust = 'flux_e_sec_{}'.format(ebv)
            ax.plot(sel['phase'], sel['flux_e_sec']/sel[fdust])

plt.show()
