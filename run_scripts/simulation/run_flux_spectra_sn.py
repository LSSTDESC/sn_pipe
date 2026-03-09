#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:23:46 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_flux import SNflux


parser = OptionParser(description='script to generate LC and spectra for SNe Ia')

parser.add_option('--x1', type=float, default=0.0,
                  help='SN Ia strech [%default]')
parser.add_option('--color', type=float, default=0.0,
                  help='SN Ia color [%default]')
parser.add_option('--daymax', type=float, default=68000,
                  help='SN Ia T0 [%default]')
parser.add_option('--z', type=float, default=0.8,
                  help='SN Ia redshift [%default]')
parser.add_option('--ebvofMW', type=float, default=0.01,
                  help='E(B-V) of MW'' [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='precipitable water vapor [mm] [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone [dobson] [%default]')
parser.add_option('--aerosol', type=float, default=0.01,
                  help='aerosol value  [%default]')

opts, args = parser.parse_args()

pp = vars(opts)

#class instance
snflux = SNflux(pp['x1'],pp['color'],pp['daymax'],pp['z'],
                pp['ebvofMW'],
                airmass=pp['airmass'],
                pwv=pp['pwv'],ozone=pp['ozone'],aerosol=pp['aerosol'])

#grab fluxes

sn_flux = snflux.get_flux()

#grab seds

sn_sed = snflux.get_sed()

print(sn_sed)

import matplotlib.pyplot as plt

bands = 'izy'

"""
idx = sn_flux['filter'].isin(bands)
sel_flux = sn_flux[idx]
"""
print(sn_flux)
mjd_min = sn_flux['phase'].min()
mjd_max = sn_flux['phase'].max()
flux_min={}
flux_max={}

for b in bands:
    idx = sn_flux['filter'] == 'LSST:'+b
    sel = sn_flux[idx]
    flux_min[b] = sel['flux'].min()
    flux_max[b] = sel['flux'].max()
    
for sed in sn_sed:
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,1,1)
    mjd = sed.meta['mjd']
    fig.suptitle('MJD:{}'.format(mjd))
    ax1.plot(sed['wavelength'],sed['flux'],'k.')
    ax1.grid(visible=True)
    ax1.set_xlim([4500.,20000.])
    for i,b in enumerate(bands):
        idx = sn_flux['filter'] == 'LSST:'+b
        idx &= sn_flux['time'] <= mjd
        sel_flux = sn_flux[idx]
        ax = fig.add_subplot(2,3,i+4)
        ax.plot(sel_flux['phase'],sel_flux['flux'],'ko')
        ax.set_xlim([mjd_min,mjd_max])
        ax.set_ylim([flux_min[b],flux_max[b]])
        ax.grid(visible=True)
    plt.show()
