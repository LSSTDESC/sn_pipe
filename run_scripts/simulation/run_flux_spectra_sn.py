#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:23:46 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_flux import SNflux
from sn_plotter_simu.plot_sn_simu import plot_flux_spectra 
from astropy.table import Table
import astropy
from sn_tools.sn_io import checkDir

parser = OptionParser(description='script to generate LC and spectra for SNe Ia')

parser.add_option('--x1', type=float, default=0.0,
                  help='SN Ia strech [%default]')
parser.add_option('--color', type=float, default=0.0,
                  help='SN Ia color [%default]')
parser.add_option('--daymax', type=float, default=68000,
                  help='SN Ia T0 [%default]')
parser.add_option('--z', type=float, default=0.8,
                  help='SN Ia redshift [%default]')
parser.add_option('--ebvofMW', type=float, default=0.01,
                  help='E(B-V) of MW'' [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='precipitable water vapor [mm] [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone [dobson] [%default]')
parser.add_option('--aerosol', type=float, default=0.01,
                  help='aerosol value  [%default]')
parser.add_option('--sed', type=int, default=0,
                  help='to estimate sn sed [%default]')
parser.add_option('--outDir', type=str, default='../sn_flux_spectra',
                  help='output directory [%default]')
parser.add_option('--outName', type=str, default='simu1',
                  help='output file name [%default]')

opts, args = parser.parse_args()

pp = vars(opts)

#create outputdir (if necessary)

checkDir(pp['outDir'])

#class instance
snflux = SNflux(pp['x1'],pp['color'],pp['daymax'],pp['z'],
                pp['ebvofMW'],
                airmass=pp['airmass'],
                pwv=pp['pwv'],ozone=pp['ozone'],aerosol=pp['aerosol'])

#grab fluxes and save output

df_flux = snflux.get_flux()
sn_flux = Table.from_pandas(df_flux)
sn_flux.meta = pp

outName_f = '{}/sn_flux_{}.hdf5'.format(pp['outDir'],pp['outName'])
astropy.io.misc.hdf5.write_table_hdf5(sn_flux, 
                                      outName_f, 
                                      path='sn_flux',
                                      append=True, serialize_meta=True)
#grab seds

if pp['sed'] == 1:
    sn_sed = snflux.get_sed()
    outName_s = '{}/sn_sed_{}.hdf5'.format(pp['outDir'],pp['outName'])
    for sed in sn_sed:
        sed.meta.update(pp)
        print(sed.meta)
        key = 'sn_sed_{}'.format(sed.meta['phase'])
        astropy.io.misc.hdf5.write_table_hdf5(sed, outName_s,path=key,
                                      append=True, serialize_meta=True)


#plot_flux_spectra(sn_flux,sn_sed)


