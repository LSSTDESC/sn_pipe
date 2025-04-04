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
labelb = 'atmos'
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(right=0.80)
atmos_trans_obsatmo.plot_atmospheric_transmission(
    plt, fig=fig, ax=ax, label=labelb, color='k', linestyle='dashed')

tel = Telescope()

for key, vals in tel.filter.items():
    tel.plot_component(key, vals, fig=fig, ax=ax,
                       color=tel.filter_colors[key],
                       label='{} band'.format(key))


ax.legend(bbox_to_anchor=(1., 0.8), ncol=1, frameon=False)
ax.set_xlim([300, 1105])

h2o = '$H_2O$'
o2 = '$O_2$'

xh2o = [935., 890., 810., 700.]
yh2o = [0.54, 0.80, 0.85, 0.83]

xo2 = [755, 685.]
yo2 = [0.30, 0.72]

for key, vals in dict(zip(xh2o, yh2o)).items():
    ax.text(key, vals, h2o, fontsize=15, color='k')

for key, vals in dict(zip(xo2, yo2)).items():
    ax.text(key, vals, o2, fontsize=15, color='k')
plt.show()
