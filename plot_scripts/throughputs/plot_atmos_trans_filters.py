#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:15:04 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_telmodel.sn_atmosphere import Atmos_Transmission
from sn_telmodel.sn_telescope import Telescope
import matplotlib.pyplot as plt

# from getObsAtmo
atmos_trans_obsatmo = Atmos_Transmission(atmos_type='obsatmo')
atmos_trans_obsatmo.load_atmosphere(aerosol=0.04)


params = {}
par_names = ['airmass', 'aerosol', 'pwv', 'oz', 'beta', 'pressure']
par_plotnames = ['am', 'aer', 'pwv', 'oz', 'beta', 'P']
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

labelb = '({})=({})'.format(ran, rbn)

fig, ax = plt.subplots(figsize=(12, 8))
atmos_trans_obsatmo.plot_atmospheric_transmission(
    plt, fig=fig, ax=ax, label=labelb, color='k', linestyle='dashed')

tel = Telescope()

for key, vals in tel.filter.items():
    tel.plot_component(key, vals, fig=fig, ax=ax,
                       color=tel.filter_colors[key],
                       label='{} band'.format(key))
plt.show()
