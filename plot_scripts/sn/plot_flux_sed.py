#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 08:43:28 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_io import load_astro_table
from sn_tools.sn_lcana import get_bands_vs_z

from sn_plotter_simu.plot_sn_simu import plot_flux_spectra
  
parser = OptionParser(description='script to plot LC and spectra for SNe Ia')

parser.add_option('--dataDir', type=str, default='../sn_flux_spectra',
                  help='data dir [%default]')
parser.add_option('--fluxFile', type=str, default='sn_flux_simu1.hdf5',
                  help='flux file [%default]')
parser.add_option('--sedFile', type=str, default='sn_sed_simu1.hdf5',
                  help='sed file [%default]')
parser.add_option('--phases', type=str, default='-10,0,20',
                  help='phases to plot [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
fluxFile = opts.fluxFile
sedFile=opts.sedFile
phases = opts.phases.split(',')
phases = list(map(float, phases))
  
sn_flux = load_astro_table('{}/{}'.format(dataDir,fluxFile))
sn_sed = load_astro_table('{}/{}'.format(dataDir,sedFile))

bands = get_bands_vs_z(sn_flux.meta['z'])
plot_flux_spectra(sn_flux,sn_sed,bands=bands,phase_to_draw=phases)
