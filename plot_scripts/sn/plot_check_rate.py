#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:18:27 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
import numpy as np
from sn_analysis.sn_calc_plot import bin_it, bin_it_mean
from sn_analysis.sn_nsn_effi import getRates

def load_data(dbDir,dbName,runType,field):
    """
    Function to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        OS name.
    runType : str
        run type (DDF/WFD).
    field : str
        Field name.

    Returns
    -------
    df : pandas df
        Loaded data.

    """
    extent = ''
    
    if runType == 'WFD':
        extent = '_nosat'
    search_path = '{}/{}/{}_spectroz{}/SN*{}*.hdf5'.format(dbDir,dbName,runType,extent,field)
    
    print('search_path',search_path)
    fis = glob.glob(search_path)
    
    df = pd.DataFrame()
    
    for fi in fis:
        dfa = pd.read_hdf(fi)
        dfa['field'] = field
        df = pd.concat((df,dfa))
        
    return df
    
def get_nsn_season(grp, norm_factor=30,cumul=False):
    """
    Function to estimate the number of SN per season

    Parameters
    ----------
    grp : pandas df
        Data to process.
    norm_factor : float, optional
        normalization factor. The default is 30.

    Returns
    -------
    None.

    """

    zmin = grp['z'].min()
    zmax = grp['z'].max()
    dz = 0.05
    bins = np.arange(zmin, zmax+dz, dz)
    
    print('nsn?',len(grp))
    nsn_obs = bin_it(grp, xvar='zmeas', norm_factor=1,
                   bins=bins, outvar='nsn_obs')
    
    print('nsn bis?',nsn_obs['nsn_obs'].sum())
    
    season_length = grp['season_length'].mean()
    survey_area = grp['survey_area'].mean()
    min_rf_phase= grp['minRFphaseQual'].mean()
    max_rf_phase = grp['maxRFphaseQual'].mean()
  
    # get snrates
    from sn_tools.sn_rate import get_nsn
    df_nsn_rate = get_nsn(rate='Hounsell', H0=70, Om0=0.3,
                          zmin=zmin, zmax=zmax-dz/2, dz=dz,
                          season_length=season_length,
                          survey_area=survey_area*norm_factor, 
                          account_for_edges=True,
                          min_rf_phase=min_rf_phase, max_rf_phase=max_rf_phase)
   
    print(df_nsn_rate[['z','nsn']])
    #df_nsn_rate['nsn'] = df_nsn_rate['nsn'].round()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8))
    nn = grp.name
    figtit = nn[0]
    figtit += '\n hpix={}'.format(int(nn[1]))
    figtit += ' season={}'.format(int(nn[2]))
    fig.suptitle(figtit)
    #ax.plot(nsn_obs['zmeas'],nsn_obs['nsn_obs'])
    vva = nsn_obs['nsn_obs']
    vvb = df_nsn_rate['nsn']
    if cumul:
        vva = np.cumsum(vva)
        vvb = vvb.cumsum()
    ax.plot(nsn_obs['zmeas'],vva,'ko',label='obs') #linestyle='solid')
    ax.plot(df_nsn_rate['z'],vvb,'b*',label='rate') #linestyle='dotted')
    
    print(df_nsn_rate['nsn'].max(),np.max(vva))
    
    print('total',vva.tolist()[-1],df_nsn_rate['nsn'].sum())
    print('boo',df_nsn_rate['nsn'])
    
    ax.grid(visible=True)
    ax.set_xlabel('$z$')
    ax.set_ylabel('$N_{SN}(z<)$')
    ax.legend()
    plt.show()
    

parser = OptionParser(
    description='Script to check rate production of SNe Ia')

parser.add_option("--dbDir", type="str", 
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_lc_coadd',
                  help="data directory [%default]")
parser.add_option("--dbName", type="str", 
                  default='baseline_v5.0.0_10yrs',
                  help="OS name [%default]")
parser.add_option("--runType", type="str", default='DDF',
                  help="type of run: DDF, WFD [%default]")
parser.add_option("--field", type="str", default='COSMOS',
                  help="field [%default]")
parser.add_option("--norm_factor", type=float, default=30,
                  help="normalisation factor[%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
field = opts.field
norm_factor = opts.norm_factor

df = load_data(dbDir, dbName, runType, field)

tt = df.groupby(['field','healpixID','season']).apply(lambda x: get_nsn_season(x,norm_factor=norm_factor))





