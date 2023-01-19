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
    tel = Telescope(airmass=airmass, aerosol=False)
    # tel.Plot_Throughputs()
    #rb = [airmass]

    for b in 'ugrizy':
        #b = 'g'
        # print(airmass, b, tel.zp(b))
        rb = [airmass]
        rb.append(b)
        rb.append(tel.zp(b))
        rb.append(tel.counts_zp(b))
        r.append(rb)

print(r)
res = np.rec.fromrecords(r, names=['airmass', 'band', 'zp', 'zp_pe'])

print(res)
np.save('zero_points_airmass.npy', res)
