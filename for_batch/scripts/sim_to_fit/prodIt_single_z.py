#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:56:34 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
from sn_tools.sn_utils import get_val
import numpy as np
import pandas as pd
import copy
    
def get_script_values(pp,dict_atmos):
    """
    Function to get script values

    Parameters
    ----------
    pp : dict
        parameters.
    dict_atmos : dict
        atmos (sigma) params.

    Returns
    -------
    dd : dict
        output dict.

    """
    
    dd = copy.deepcopy(pp)    
    
    dd.pop('config',None)
    dd.pop('simuFileDir',None)
    dd.pop('atmosFile',None)
    
    for key, vals in dict_atmos.items():
        dd['sigma_{}'.format(key)] = vals
    
    
    return dd
    
    
def get_atmos_params(atmosFile,config):
    """
    Function to grab atmos parameters

    Parameters
    ----------
    atmosFile : str
        atmos file (full path).
    config : str
        config to choose.

    Returns
    -------
    dd : dict
        atmos parameters.

    """
    
    #load atmos params
    df_atm = pd.read_csv(atmosFile)

    idx = df_atm['config'] == config
    
    sel_atm = df_atm[idx]
    
    dd = {}
    if len(sel_atm) == 1:
        
        dd = sel_atm.iloc[0].to_dict()
        
    dd.pop('config',None)
    return dd
    
def process_single(pp,dict_atmos):
    """
    Processing from (z,seasons) values
    
        Parameters
        ----------
        pp : dict
            params.
        dict_atmos : dict
            atmos (sigma) params.
    
        Returns
        -------
        None.    

    
    """
    
    z = pp['zmin']
    config = pp['config']
    obs_coadd = pp['obs_coadd']
    lc_coadd = pp['lc_coadd']
    
    procName='sn_z_single_{}_{}_{}_{}'.format(np.round(z,2),config,
                                              obs_coadd,lc_coadd)

    bb = BatchIt(processName=procName)

    script = 'run_scripts/sim_to_fit/sim_to_fit_single.py'
    ppb = copy.deepcopy(pp)
    for seas in get_val(pp['seasons']):
        ppb['seasons'] = seas
        dd = get_script_values(ppb,dict_atmos)
        #print(dd)
        bb.add_batch(script,dd)
    
    bb.go_batch()

def process_from_simu(pp,dict_atmos):
    """
    Processing from simu parameter file

    Parameters
    ----------
    pp : dict
        params.
    dict_atmos : dict
        atmos (sigma) params.

    Returns
    -------
    None.

    """
    
    simDir = pp['simuFileDir']
    config = pp['config']
    obs_coadd = pp['obs_coadd']
    lc_coadd = pp['lc_coadd']
    dbName = pp['dbName']
    conf = pp['config']
    
    z = float(simDir.split('/')[-1].split('_')[-1])
    z = np.round(z,2)
    pp['zmin'] = z
    pp['outDir'] = '{}/{}_z_{}_{}_{}'.format(pp['outDir'],conf,z,
                                             obs_coadd,lc_coadd)
    procName='sn_z_single_from_simu_{}_{}_{}_{}_{}'.format(conf,z,
                                                        config,
                                                        obs_coadd,lc_coadd)

    bb = BatchIt(processName=procName)

    script = 'run_scripts/sim_to_fit/sim_to_fit_single.py'
    
    import glob
    fis = glob.glob('{}/{}/DDF_spectroz/*.hdf5'.format(simDir,dbName))

    ppb = copy.deepcopy(pp)
    for fi in fis:
        #print(fi)
        seas = fi.split('.hdf5')[0].split('/')[-1].split('_')[-1]
        #print(seas)
        ppb['seasons'] = seas
        ppb['SN_simuFile'] = fi
        dd = get_script_values(ppb,dict_atmos)
        #print('oooo',dd)
        bb.add_batch(script,dd)
        
    bb.go_batch()
    
parser = OptionParser(
    description='Script to produce SN for WFD and DDF surveys for a pixel')

parser.add_option("--outDir", type="str",
                  default='../totoooo',
                  help="Main output dir [%default]")
parser.add_option("--simuFileDir", type=str,
                  default='None',
                  help="simuFile Dir [%default]")
parser.add_option("--seasons", type=str,
                  default='1-10',
                  help="seasons to process [%default]")
parser.add_option("--zmin", type=float,
                  default=0.01,
                  help="redshift to process [%default]")
parser.add_option("--dbDir", type=str,
                  default='../DB_Files',
                  help="DB location dir [%default]")
parser.add_option("--dbName", type=str,
                  default='baseline_v5.3.0_10yrs',
                  help="OS to process [%default]")
parser.add_option("--dbExtens", type=str,
                  default='npy',
                  help="DB file extensions [%default]")
parser.add_option("--config", type=str,
                  default='confe',
                  help="atmos config [%default]")
parser.add_option("--atmosFile", type=str,
                  default='input/zp_atmos/config_atmos.csv',
                  help="atmos config file [%default]")
parser.add_option("--obs_coadd", type=int,
                  default=0,
                  help="coadd obs [%default]")
parser.add_option("--lc_coadd", type=int,
                  default=1,
                  help="coadd lc [%default]")
parser.add_option("--nsn_abs", type=int,
                  default=300,
                  help="NSN to generate [%default]")
parser.add_option("--fit_lc", type=int,
                  default=1,
                  help="to fit lcs [%default]")
parser.add_option("--save_LC", type=int,
                  default=0,
                  help="to save lcs [%default]")
parser.add_option("--x1_type", type=str,
                  default='random',
                  help="x1 type values [%default]")
parser.add_option("--color_type", type=str,
                  default='random',
                  help="color type values [%default]")
parser.add_option("--z_type", type=str,
                  default='unique',
                  help="z type value[%default]")

opts, args = parser.parse_args()

pp = vars(opts)

#get atmos parameters
dict_atmos = get_atmos_params(pp['atmosFile'], pp['config'])

print(dict_atmos)

if pp['simuFileDir'] == 'None':
    process_single(pp,dict_atmos)
else:
    process_from_simu(pp,dict_atmos)


    
    
    
    


