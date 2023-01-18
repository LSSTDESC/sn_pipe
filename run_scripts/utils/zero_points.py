#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:10:52 2023

@author: gris
"""

from sn_tools.sn_telescope import Telescope
import numpy as np

r = []
for airmass in np.arange(1., 2.501, 0.01):
    tel = Telescope(airmass=airmass)
    rb = [airmass]
    for b in 'ugrizy':
        #print(airmass, b, tel.zp(b))
        rb.append(tel.zp(b))
    r.append(rb)

res = np.rec.fromrecords(r, names=['airmass', 'u', 'g', 'r', 'i', 'z', 'y'])

print(res)
np.save('zero_points_airmass.npy', res)
