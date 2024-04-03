#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:14:53 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_telmodel.sn_telescope import get_telescope
from sn_telmodel import plt
from optparse import OptionParser

parser = OptionParser(description='Script to plot telescope throughputs')

parser.add_option('--tel_dir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag versions of the throughputs [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=str, default='aerosol',
                  help='aerosol value [%default]')

opts, args = parser.parse_args()

tel_dir = opts.tel_dir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
airmass = opts.airmass
tag = opts.tag
aerosol = opts.aerosol


telb = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(telb, throughputsDir)
atmos_dir = '{}/{}'.format(telb, atmosDir)
tel = get_telescope(tel_dir=telb,
                    through_dir=through_dir,
                    atmos_dir=atmos_dir,
                    tag=tag, airmass=airmass, aerosol=aerosol)

tel.plot_Throughputs()

plt.show()
