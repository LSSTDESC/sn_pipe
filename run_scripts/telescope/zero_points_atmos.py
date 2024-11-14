#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_telmodel.sn_telescope import get_telescope
import numpy as np

tel_dir = 'throughputs'
through_dir = 'baseline'

airmass = 2.0
pwv = 4.0
ozone = 300.
beta = 1.4
aerosol = 0.

sigma_airmass = 0.01
sigma_pwv = 0.2
sigma_ozone = 10.
tag = '1.9'

tel_dir = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(tel_dir, through_dir)


for i in range(3):
    airm = airmass+np.random.normal(0., sigma_airmass)
    oz = ozone+np.random.normal(0., sigma_ozone)
    prec_wp = pwv+np.random.normal(0., sigma_pwv)
    tel = get_telescope(tel_dir=tel_dir,
                        through_dir=through_dir,
                        tag=tag, load_components=True,
                        airmass=airm,
                        aerosol=aerosol, pwv=prec_wp, oz=oz)

    tel.mean_wave()
    for b in 'ugrizy':
        mean_wave = tel.mean_wavelength[b]
        zp = tel.zp(b)
        print(i, b, mean_wave, zp)
