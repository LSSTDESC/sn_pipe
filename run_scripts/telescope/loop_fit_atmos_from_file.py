#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:10:28 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import os


airmasses = np.arange(1.0, 2.6, 0.1)

for airmass in airmasses:
    vv = np.round(airmass, 1)
    fName = 'airmass_{}.hdf5'.format(airmass)
    cmd = 'python run_scripts/telescope/fit_atmos_from_file.py '
    cmd += '--airmass={} --outName=airmass_{}.hdf5'.format(vv, vv)
    print(cmd)
    os.system(cmd)
