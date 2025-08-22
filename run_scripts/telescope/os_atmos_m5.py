#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:02:11 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
from sn_telmodel.sn_throughputs import get_telescope
import pandas as pd

parser = OptionParser(
    description='Script to estimate m5 from atmos parameters')

parser.add_option('--fileDir', type=str, default='../DB_Files',
                  help='OS file dir [%default]')
parser.add_option('--dbName', type=str, default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--tel_dir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.04,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=5.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=320.,
                  help='ozone value [%default]')
parser.add_option('--gain', type=float, default=2.5,
                  help='electronic gain [%default]')
parser.add_option('--pressure', type=float, default=743.,
                  help='pressure on the Cerro Pachon [%default]')
parser.add_option('--atmos_type', type=str, default='obsatmo',
                  help='how is the atmos estimated (obsatmo, from_file) [%default]')
opts, args = parser.parse_args()

fileDir = opts.fileDir
dbName = opts.dbName
tag = opts.tag
tel_dir = opts.tel_dir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
airmass = opts.airmass
pwv = opts.pwv
ozone = opts.ozone
aerosol = opts.aerosol
gain = opts.gain
pressure = opts.pressure
atmos_type = opts.atmos_type

# throughputs instance
telb = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(telb, throughputsDir)
atmos_dir = '{}/{}'.format(telb, atmosDir)
throughput = get_telescope(tel_dir=telb,
                           through_dir=through_dir,
                           atmos_dir=atmos_dir,
                           atmos_type=atmos_type,
                           tag=tag, load_components=True,
                           airmass=airmass, aerosol=aerosol,
                           pwv=pwv, ozone=ozone, gain=gain, pressure=pressure)

# grab OS data
fName = '{}/{}.npy'.format(fileDir, dbName)
os_data = np.load(fName)

idx = os_data['moonPhase'] < 0.5
os_data = os_data[idx]
io = 0
for x in os_data:
    print(x.dtype.names)
    io += 1
    throughput.reset_data()
    b = x['filter']
    fwhmeff = x['seeingFwhmEff']
    m5 = x['fiveSigmaDepth']
    exptime = x['exptime']
    nexp = x['numExposures']
    airmass = x['airmass']
    sky = x['sky']
    moonPhase = x['moonPhase']

    throughput.data['FWHMeff'][b] = fwhmeff
    throughput.load_atmosphere(
        airmass=airmass, pwv=pwv, ozone=ozone, aerosol=aerosol)

    sky_new = throughput.mag_sky(b)
    m5_new = throughput.m5(b, exptime, nexp)

    print(b, airmass, fwhmeff, m5, m5_new, m5-m5_new, sky-sky_new, moonPhase)

    print('allo', throughput.data['FWHMeff'])
    # throughput.data['FWHMeff'][b] = fwhmeff
    throughput.etc()

    if io > 2:
        break
