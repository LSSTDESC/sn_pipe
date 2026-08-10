#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:43:09 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from optparse import OptionParser
import os


parser = OptionParser(
    description='Script to loop produce SN for WFD and DDF surveys for a pixel')

parser.add_option("--z_min", type=float,
                  default=0.01,
                  help="zmin [%default]")
parser.add_option("--z_max", type=float,
                  default=1.1,
                  help="zmax [%default]")
parser.add_option("--z_step", type=float,
                  default=0.01,
                  help="zstep value [%default]")
parser.add_option("--dbDir", type=str,
                  default='../DB_Files',
                  help="DB location dir [%default]")
parser.add_option("--dbName", type=str,
                  default='baseline_v5.3.0_10yrs',
                  help="OS to process [%default]")
parser.add_option("--dbExtens", type=str,
                  default='npy',
                  help="DB file extensions [%default]")


opts, args = parser.parse_args()

params = vars(opts)

z_min = opts.z_min
z_max = opts.z_max
z_step = opts.z_step

zvals = np.arange(0,z_max+z_step,z_step)
    
idx = zvals >= z_min
zvals = zvals[idx]

idxb = zvals == z_min
sel = zvals[idxb]

if len(sel) == 0:
    zvals = np.append(zvals,z_min)
    
zvals = np.sort(zvals)

idx = zvals <= z_max

zvals = zvals[idx]

print('list of z to process',zvals,len(zvals))

script = "python for_batch/scripts/sim_to_fit/prodIt_single_z.py"

vvals = ['dbDir','dbName','dbExtens']

for zz in zvals:
    scr_ = '{} --z={}'.format(script,np.round(zz,2))
    for vv in vvals:
        scr_ += ' --{}={}'.format(vv,params[vv])
    os.system(scr_)
    
    