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

def get_script_values(season,z):
    """
    build a dict of script values

    Parameters
    ----------
    season : int
        season of observation.
    z : float
        redshift.

    Returns
    -------
    dd : dict
        script parameter values.

    """
    
    zval = np.round(z,2)
    dd = {}
    
    dd['season'] = season
    dd['x1_type']='random'
    dd['color_type'] = 'random'
    dd['z_type'] = 'unique'
    dd['zmin'] = '{}'.format(zval)
    dd['nsn_abs'] = 300
    dd['outDir'] = '../ref_LC_{}'.format(zval)
    dd['smear_flux'] = 1
    dd['save_LC'] = 0
    dd['obs_coadd'] = 1   
    dd['lc_coadd'] = 0
    dd['fit_lc'] = 1
    
    return dd
   


parser = OptionParser(
    description='Script to produce SN for WFD and DDF surveys for a pixel')

parser.add_option("--outDir_main", type="str",
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu_single',
                  help="Main output dir [%default]")
parser.add_option("--seasons", type=str,
                  default='1-10',
                  help="seasons to process [%default]")
parser.add_option("--z", type=float,
                  default=0.01,
                  help="redshift to process [%default]")

opts, args = parser.parse_args()

seasons = get_val(opts.seasons)
z = opts.z

procName='sn_z_single_{}'.format(np.round(z,2))

bb = BatchIt(processName=procName)

script = 'run_scripts/sim_to_fit/sim_to_fit_single.py'
for seas in seasons:
    dd = get_script_values(seas,z)
    #print(dd)
    bb.add_batch(script,dd)
    
bb.go_batch()


