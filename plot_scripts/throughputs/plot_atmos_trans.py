#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:15:04 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from sn_telmodel.sn_atmosphere import Atmos_Transmission
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser(description='Script to plot a&tmos transmission')

parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--airmass', type=float, default=1.0,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.0,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone value [%default]')

opts, args = parser.parse_args()

atmosDir = 'throughputs_{}/{}'.format(opts.tag, opts.atmosDir)
airmass = opts.airmass
aerosol = opts.aerosol
pwv = opts.pwv
ozone = opts.ozone

# from file
atmos_trans_file = Atmos_Transmission(
    atmos_dir=atmosDir, atmos_type='from_file')
atmos_trans_file.load_atmosphere(airmass=airmass,atmos_type='from_file')

# from getObsAtmo
atmos_trans_obsatmo = Atmos_Transmission(atmos_type='obsatmo')
atmos_trans_obsatmo.load_atmosphere(
    airmass=airmass, pwv=pwv, ozone=ozone, aerosol=aerosol)


params = {}
par_names = ['airmass', 'aerosol', 'pwv', 'ozone', 'beta', 'pressure']
par_plotnames = ['am', 'aer', 'pwv', 'ozone', 'beta', 'P']
pars = dict(zip(par_names, par_plotnames))
for key, vals in pars.items():
    sstr = 'params[\'{}\'] = atmos_trans_obsatmo.{}'.format(vals, key)
    exec(sstr)

ra = []
rb = []
for key, vals in params.items():
    ra.append(key)
    rb.append(vals)

ran = ','.join(ra)
rb = list(map(str, rb))
rbn = ','.join(rb)

# superimpose atmospheric transmission curves
labela = 'from file airmass({})+aerosol'.format(atmos_trans_file.airmass)
labelb = '({})=({})'.format(ran, rbn)

fig, ax = plt.subplots(figsize=(12, 8))
atmos_trans_file.plot_atmospheric_transmission(
    plt, fig=fig, ax=ax, label=labela)
atmos_trans_obsatmo.plot_atmospheric_transmission(
    plt, fig=fig, ax=ax, label=labelb, color='k', linestyle='dashed')

"""
# residuals
vva = atmos_trans_file.atmosphere.sb
vvb = atmos_trans_obsatmo.atmosphere.sb

figb, axb = plt.subplots(figsize=(12, 8))
axb.plot(atmos_trans_file.atmosphere.wavelen, vva-vvb)

print(np.sum((vva-vvb)**2))
"""

plt.show()
