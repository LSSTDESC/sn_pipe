#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:45:38 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import os
import numpy as np


x1 = 0.0
color = 0.

script = 'python run_scripts/simulation/run_flux_spectra_sn.py'
script += ' --x1={}'.format(x1)
script += ' --color={}'.format(color)
outDir = '../sn_flux_z_airmass'
script += ' --outDir={}'.format(outDir)

z = np.arange(0.0,1.2,0.1)
airmass = np.arange(1.0,3.,0.1)
nfi = 0
for i, vv in enumerate(z):
    if vv < 0.01:
        vv = 0.01
    vv = np.round(vv,2)
    for airm in airmass:
        nfi +=1
        outName = 'simu_{}'.format(nfi)
        scr_ = script
        scr_ += ' --z={}'.format(vv)
        scr_ += ' --outName={}'.format(outName)
        scr_ += ' --airmass={}'.format(airm)
        print(scr_)
        os.system(scr_)