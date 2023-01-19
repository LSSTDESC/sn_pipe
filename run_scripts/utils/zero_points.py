#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:10:52 2023

@author: gris
"""

from sn_tools.sn_telescope import Telescope
import numpy as np

r = []
for airmass in np.arange(1., 2.51, 0.1):
    # for airmass in [1.2]:
    tel = Telescope(airmass=airmass, aerosol=True)
    # tel.Plot_Throughputs()
    #rb = [airmass]

    for b in 'ugrizy':
        #b = 'g'
        # print(airmass, b, tel.zp(b))
        mean_wave = tel.mean_wavelength[b]
        rb = [airmass]
        rb.append(b)
        rb.append(tel.zp(b))
        rb.append(tel.counts_zp(b))
        rb.append(mean_wave)
        r.append(rb)

print(r)
res = np.rec.fromrecords(
    r, names=['airmass', 'band', 'zp', 'zp_adu_sec', 'mean_wavelength'])

print(res)
np.save('zero_points_airmass.npy', res)
