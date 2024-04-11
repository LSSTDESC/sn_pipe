#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:03:33 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import sn_simu_input as simu_input
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_io import add_parser, checkDir
from sn_tools.sn_process import Process
from optparse import OptionParser
import sn_script_input
from sn_simu_wrapper.sn_wrapper_for_simu import SimuWrapper

path_process_input = sn_script_input.__path__
path_simu = simu_input.__path__
confDict_simu = make_dict_from_config(path_simu[0], 'config_simulation.txt')
confDict_gen = make_dict_from_config(
    path_process_input[0], 'config_process.txt')
parser = OptionParser()
# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict_gen)
add_parser(parser, confDict_simu)

opts, args = parser.parse_args()

simuDict = {}
procDict = {}
for key, vals in confDict_simu.items():
    # simuDict[key] = eval('opts.{}'.format(key))
    newval = eval('opts.{}'.format(key))
    simuDict[key] = (vals[0], newval)
for key, vals in confDict_gen.items():
    procDict[key] = eval('opts.{}'.format(key))
# new dict with configuration params
yaml_params = make_dict_from_optparse(simuDict)

# one modif: full dbName
yaml_params['Observations']['filename'] = '{}/{}.{}'.format(
    opts.dbDir, opts.dbName, opts.dbExtens)

# create outputdir if does not exist
outDir_simu = yaml_params['OutputSimu']['directory']

prodid = yaml_params['ProductionIDSimu']

metricList = [SimuWrapper(yaml_params)]
fieldType = yaml_params['Observations']['fieldtype']
fieldName = yaml_params['Observations']['fieldname']
nside = yaml_params['Pixelisation']['nside']
saveData = 0
outDir = yaml_params['OutputSimu']['directory']
# now perform the processing


# print('seasons and metric', opts.Observations_season,
#      metricList, opts.pixelmap_dir, opts.npixels)

procDict['fieldType'] = opts.fieldType
procDict['metricList'] = metricList
procDict['fieldName'] = opts.fieldName
procDict['outDir'] = outDir
procDict['pixelList'] = opts.pixelList
procDict['nside'] = opts.nside

# print('processing', procDict)
process = Process(**procDict)
