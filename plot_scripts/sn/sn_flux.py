#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:08:10 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import glob
from sn_tools.sn_io import load_astro_table
from sn_tools.sn_lcana import get_bands_vs_z
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pandas as pd

def get_flux_sum_band(grp,exptime=30):
    """
    Function to estimate the integrated flux per band

    Parameters
    ----------
    grp : pandas df
        Data to process.
    exptime : float, optional
        exposure time. The default is 30.

    Returns
    -------
    pandas df
        flux and err.

    """
    
    phase_min = grp['phase'].min()
    phase_max = grp['phase'].max()
    
    grp['flux_int'] = grp['flux']*exptime
    def integrand(p):
        ph = grp['phase'].tolist()
        flux = grp['flux_int'].tolist()
        myinterp = interp1d(ph,flux,bounds_error=False, fill_value=0.)
        
        return myinterp(p)
    
    res,abserr = quad(integrand, phase_min, phase_max)[:2]
    dictout = {}
    dictout['flux'] = [res]
    dictout['fluxerr'] = [abserr]

    df = pd.DataFrame.from_dict(dictout)

    return df
    


def get_flux_sum_deprecated(tab,z,exptime=30.):
    
    #grab the bands of interest
    bands = get_bands_vs_z(z)
    
    idx = np.in1d(tab['filter_notel'],list(bands))
    idx &= tab['flux'] > 0
    
    sel = tab[idx]
    sel['flux_int'] = sel['flux']*exptime
    
    r = []
    for b in bands:
        idx = sel['filter_notel'] == b
        grp = sel[idx]
        dphase = np.mean(np.diff(grp['phase']))
        print(b, dphase,np.sum(grp['flux_int']*dphase))
        phase_min = grp['phase'].min()
        phase_max = grp['phase'].max()
        
        def integrand(p):
            ph = grp['phase'].tolist()
            flux = grp['flux_int'].tolist()
            myinterp = interp1d(ph,flux,bounds_error=False, fill_value=0.)
            
            return myinterp(p)
        res,abserr = quad(integrand, phase_min, phase_max)[:2]
        print(b,res,abserr)
        r.append([b,res,abserr,z])

    df = pd.DataFrame(r, columns=['band','flux','fluxerr'])
    return df

def process_data(z,fis):
    """
    Function to process data

    Parameters
    ----------
    z : float
        Redshift value.
    fis : list(str)
        List of files to (potentially) process.

    Returns
    -------
    dft : pandas df
        Processed data.

    """
    
    dft = pd.DataFrame()
    colby = 'filter_notel'
    for fi in fis:
        print('loading',fi)
        zval = fi.split('/')[-1].split('_')[3]
        if zval == str(z):
            tab = load_astro_table(fi)
            print(tab.meta)
            df = tab.to_pandas()
            bands = get_bands_vs_z(z)
            idx = df['filter_notel'].isin(list(bands))
            
            dfr = df[idx].groupby(colby).apply(lambda x: get_flux_sum_band(x),
                                           include_groups=False).reset_index()
            dfr['airmass'] = tab.meta['airmass']
            dfr['z'] = tab.meta['z']
            dfr['x1'] = tab.meta['x1']
            dfr['color'] = tab.meta['color']
            dfr['daymax'] = tab.meta['daymax']
            dft = pd.concat((dft,dfr))
            
    return dft

parser = OptionParser(description='analyze and plot of LC flux files')

parser.add_option('--fluxDir', type=str, default='../sn_flux_z_airmass',
                  help='data dir [%default]')

opts, args = parser.parse_args()

fluxDir = opts.fluxDir

fis = glob.glob('{}/*.hdf5'.format(fluxDir))

#loop on files and grab tables
z = np.arange(0.0,1.2,0.1)

df = pd.DataFrame()

for zv in z:
    if zv < 0.01:
        zv = 0.01
    vv = np.round(zv,2)
    dfa = process_data(vv,fis)
    df = pd.concat((df,dfa))
        
print(df)
    
