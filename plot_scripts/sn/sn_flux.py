#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:08:10 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import glob

parser = OptionParser(description='analyze and plot of LC flux files')

parser.add_option('--fluxDir', type=str, default='../sn_flux_z',
                  help='data dir [%default]')

opts, args = parser.parse_args()

fluxDir = opts.fluxDir

fis = glob.glob('{}/*.hdf5'.format(fluxDir))

print(fis)

#loop on files and grab tables

for fi in fis:
    print('loading',fi)