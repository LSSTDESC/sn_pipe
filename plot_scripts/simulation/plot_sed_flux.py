#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:41:26 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import h5py
#from astropy.table import Table, vstack
from optparse import OptionParser
from sn_plotter_simu.plot_sn_simu import plot_flux_spectra
#import astropy
from sn_tools.sn_io import checkDir,load_astro_table



def load_sed(fName):
    """
    Function to load sn SED

    Parameters
    ----------
    fName : str
        file name.

    Returns
    -------
    r : list(astropy table)
        list of SEDs.

    """
    
    fFile = h5py.File(fName, 'r')
    keys = list(fFile.keys())
    
    r = []
    for key in keys:
        tab = get_table(fFile, key)
        if tab.meta:
            r.append(tab)
        
    return r
    
    
    

parser = OptionParser(description='display SN SED and corresponding flux')

parser.add_option("--fileDir", type="str", default='../sn_flux_spectra',
                  help="file directory [%default]")
parser.add_option("--fileName", type="str", default='simu1',
                  help="file name [%default]")
parser.add_option("--outDir", type="str", default='None',
                  help="output dir [%default]")

opts, args = parser.parse_args()

fDir = opts.fileDir
fName = opts.fileName
outDir = opts.outDir

if outDir != 'None':
    checkDir(outDir)

file_flux = '{}/sn_flux_{}.hdf5'.format(fDir,fName)
file_sed = '{}/sn_sed_{}.hdf5'.format(fDir,fName)

sn_flux = load_astro_table(file_flux)

sn_sed = load_astro_table(file_sed)

print(sn_flux)
print(sn_sed)
phase_to_draw = [-10,0.,10]
plot_flux_spectra(sn_flux,sn_sed,outDir=outDir,phase_to_draw=phase_to_draw)
