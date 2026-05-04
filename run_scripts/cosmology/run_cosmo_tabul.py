#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:02:31 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
#from sn_tools.sn_io import checkDir
from sn_cosmology.cosmo_tabul import Cosmo_tabul
from sn_tools.sn_interp import RegularGrid_interp
from sn_cosmology.cosmo_tabul import check_interp
    
parser = OptionParser(description='Script to estimate distmod in 3D')

parser.add_option('--deparams', type=str,
                  default='w1,w2',
                  help='DE eos parameters [%default]')
parser.add_option('--cosmofitparams', type=str,
                  default='w1,w2,Om0',
                  help='parameters used to estimate distmod [%default]')
parser.add_option('--cosmofitparams_min', type=str,
                  default='-1.,-10.,0.2',
                  help='fit parameter min values [%default]')
parser.add_option('--cosmofitparams_max', type=str,
                  default='0.,-1.,0.4',
                  help='fit parameter max values [%default]')
parser.add_option('--cosmofitparams_delta', type=str,
                  default='0.1,0.1,0.05',
                  help='fit parameter n values [%default]')
parser.add_option('--declass', type=str,
                  default='DDE_FLRW',
                  help='DE class to use (w0waCDM/DDE_FLRW) [%default]')
parser.add_option('--classloc', type=str,
                  default='sn_tools.sn_cosmo_model',
                  help='DE class location \
                        (astropy.cosmology/sn_tools.sn_cosmo_model) [%default]')
parser.add_option('--demodel', type=str,
                  default='oscilla',
                  help='DE eos model name [%default]')
parser.add_option('--deeos', type=str,
                  default='-1.+(w1*z*np.sin(w2*z))/(1+z**2)',
                  help='DE eos model [%default]')
parser.add_option('--H0', type=float,
                  default=70.,
                  help='DE eos model [%default]')
parser.add_option('--Om0', type=float,
                  default=0.30,
                  help='Omega_matter [%default]')
parser.add_option('--Ode0', type=float,
                  default=0.70,
                  help='Omega_DE [%default]')
parser.add_option('--outName', type=str,
                  default='distmod_tabul',
                  help='prefix for output name [%default]')
parser.add_option('--outDir', type=str,
                  default='../distmod_tabul',
                  help='output directory [%default]')

opts, args = parser.parse_args()

"""
checkDir(opts.outDir)
outName = '{}/{}_{}.hdf5'.format(opts.outDir,opts.outName,opts.demodel)
"""
params = vars(opts)

# cosmo tabul
costab = Cosmo_tabul(params)
df_tot = costab()

print(df_tot)

#save the data
if opts.outDir != 'None':
    from sn_tools.sn_io import checkDir
    checkDir(opts.outDir)
    outName = '{}/{}_{}.hdf5'.format(opts.outDir,opts.outName,opts.demodel)
    df_tot.to_hdf(outName,key='distmod')

#get interpolator
ccols = params['cosmofitparams'].split(',')
ccols.append('z')

interpa = RegularGrid_interp(df_tot,ccols) 

interp = interpa()
#cross check

check_interp(interp,df_tot,ccols,params)