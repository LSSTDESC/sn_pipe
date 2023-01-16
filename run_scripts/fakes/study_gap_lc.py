#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:28:06 2023

@author: Philippe Gris
"""

from sn_simu_wrapper.sn_wrapper_for_simu import InfoFitWrapper
import sn_fit_input as simu_fit
import sn_script_input
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse, add_parser
from optparse import OptionParser
import os
import yaml
from astropy.table import Table
import numpy as np
import h5py
import random

def load_lc_as_list(fName):
   """
    Load LC as list of astropy table
    from file

    Parameters
    ----------
    fName : TYPE
        DESCRIPTION.

    Returns
    -------
    lc : TYPE
        DESCRIPTION.

   """
    
   f = h5py.File(fName, 'r')
   keys = list(f.keys())

   lc = []
   for key in keys:
       tt = Table.read(fName, path=key)
       lc.append(tt)
   return lc
    


path_fit = simu_fit.__path__
path_process_input = sn_script_input.__path__
confDict_info = make_dict_from_config(
    path_process_input[0], 'config_sel.txt')
confDict_fit = make_dict_from_config(path_fit[0], 'config_fit.txt')

parser = OptionParser()

parser = OptionParser()
# parser for simulation parameters : 'dynamical' generation

add_parser(parser, confDict_info)
add_parser(parser, confDict_fit)

opts, args = parser.parse_args()

infoDict = {}
fitDict = {}

for key, vals in confDict_info.items():
    infoDict[key] = eval('opts.{}'.format(key))
# load the new values
for key, vals in confDict_fit.items():
    newval = eval('opts.{}'.format(key))
    fitDict[key] = (vals[0], newval)
    
 
yaml_params_fit = make_dict_from_optparse(fitDict)

outDir = yaml_params_fit['Simulations']['dirname']
prodid = yaml_params_fit['Simulations']['prodid']

if not os.path.isdir(outDir):
    os.makedirs(outDir)
    
yaml_name_fit = '{}/{}_fit.yaml'.format(outDir, prodid)
with open(yaml_name_fit, 'w') as f:
    data_fit = yaml.dump(yaml_params_fit, f)


process = InfoFitWrapper(infoDict,yaml_params_fit)

# load initial lc file
import glob

search_path = '{}/LC_{}*.hdf5'.format(outDir,prodid)
print('searching...',search_path)
fis = glob.glob(search_path)

# load Light curves in list
lc_list = []

for fi in fis:
    lc_list += load_lc_as_list(fi)

print('loaded',len(lc_list))
   
nights = []
for lc in lc_list:
    if np.abs(lc.meta['z']-1.11) < 1.e-5:
        print('ici',lc.meta['z'],list(np.unique(lc['night'])))    
        nights = list(np.unique(lc['night']))

from math import comb       
for i in range(1,len(nights)):
    print(comb(len(nights),i))
    rc = random.choices(nights,k=i)
    print(rc)

"""
res = process.run(lc_list)

print(res)
"""