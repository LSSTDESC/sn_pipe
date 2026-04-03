#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:23:46 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_flux import SNflux
from astropy.table import Table,vstack
import astropy
from sn_tools.sn_io import checkDir
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_tools.sn_lcana import get_bands_vs_z
import copy
import os

def get_dict_for_class(prodDict):
    """
    Function to generate a dict for the SNflux class params

    Parameters
    ----------
    prodDict : dict
        input dict.

    Returns
    -------
    dict
        output dict.

    """
    params = copy.deepcopy(procDict)
    
    #drop unnecessary for snflux
    for vv in ['sed', 'outDir','outName','outDirDisplay']:
        del params[vv]
    
    #some modif of the cosmo part
    de_values = params['de_values'].split(',')
    de_params = params['de_params'].split(',')
    
    de_values = list(map(float,de_values))
    ppn = {}
    ppn['de_params'] = dict(zip(de_params,de_values))
    
    for vv in ['de_values','de_class','de_model','de_eos','H0','Om0','Ode0','class_loc']:
        ppn[vv] = params[vv]
        del params[vv]
    del params['de_params']
    
    params['cosmo_params'] = ppn

    return copy.deepcopy(params)


    
parser = OptionParser(description='script to generate LC and spectra for SNe Ia')

confDict = make_dict_from_config('input_script', 'config_sn_flux_spectra.txt')

add_parser(parser, confDict)

opts, args = parser.parse_args()

pp = vars(opts)

#create outputdir (if necessary)

checkDir(pp['outDir'])

procDict = {}
for key, vals in confDict.items():
    # simuDict[key] = eval('opts.{}'.format(key))
    newval = eval('opts.{}'.format(key))
    #procDict[key] = (vals[0], newval)
    procDict[key] = newval
 

params = get_dict_for_class(procDict)

if params['phases_sed'] == 'None':
    params['phases_sed'] = []
else:
    params['phases_sed'] = list(map(float,params['phases_sed'].split(',')))

#class instance
snflux = SNflux(**params)

#grab fluxes and save output

df_flux = snflux.get_flux()
sn_flux = Table.from_pandas(df_flux)
sn_flux.meta = pp


if pp['outName'] != 'None':
    outName_f = '{}/sn_flux_{}.hdf5'.format(pp['outDir'],pp['outName'])
    #if the file already exist: remove it!
    if os.path.isfile(outName_f):
        os.system('rm {}'.format(outName_f))
    #save data here
    astropy.io.misc.hdf5.write_table_hdf5(sn_flux, 
                                          outName_f, 
                                          path='sn_flux',
                                          append=True, 
                                          serialize_meta=True,
                                          overwrite=True)
#grab seds

if pp['sed'] == 1:
    sn_sed = snflux.get_sed()
   
    tab = Table()
    for sed in sn_sed:
        """
        sed.meta.update(pp)
        print(sed.meta)
        key = 'sn_sed_{}'.format(sed.meta['phase'])
        astropy.io.misc.hdf5.write_table_hdf5(sed, outName_s,path=key,
                                      append=True, serialize_meta=False,
                                      overwrite=True)
        """
        sed['phase'] = sed.meta['phase']
        sed['mjd'] = sed.meta['mjd']
        
        tab = vstack([tab,sed],metadata_conflicts='silent')
        
    tab.meta = pp
    if pp['outName'] != 'None':
        outName_s = '{}/sn_sed_{}.hdf5'.format(pp['outDir'],pp['outName'])
        #if the file already exist: remove it!
        if os.path.isfile(outName_s):
            os.system('rm {}'.format(outName_s))
        #save data here
        astropy.io.misc.hdf5.write_table_hdf5(tab, outName_s,path='sn_sed',
                                              append=True, serialize_meta=True,
                                              overwrite=True)
    
if pp['outDirDisplay'] != 'None':
    #check if outdir already exist: if yes, remove it!
    if os.path.exists(pp['outDirDisplay']):
        os.system('rm -rf {}'.format(pp['outDirDisplay']))    
    
    checkDir(pp['outDirDisplay'])
    from sn_plotter_simu.plot_sn_simu import plot_flux_spectra
    bands = get_bands_vs_z(pp['z'])
    plot_flux_spectra(sn_flux,tab,outDir=pp['outDirDisplay'],bands=bands)


