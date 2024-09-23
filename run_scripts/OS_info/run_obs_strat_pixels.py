#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:50:42 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import sn_script_input
import sn_metrics_input
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_io import add_parser
from optparse import OptionParser
import yaml
from sn_metrics.sn_obs_strat_pixels import SNObsStratPixel

# get all possible simulation parameters and put in a dict
path_process_input = sn_script_input.__path__
path_obstrat_input = sn_metrics_input.__path__

confDict_gen = make_dict_from_config(
    path_process_input[0], 'config_process.txt')
confDict_obstrat = make_dict_from_config(
    path_obstrat_input[0], 'config_SN_ObsStratPixels.txt')

parser = OptionParser()

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict_gen)
add_parser(parser, confDict_obstrat)

opts, args = parser.parse_args()

procDict = {}
for key, vals in confDict_gen.items():
    procDict[key] = eval('opts.{}'.format(key))
metDict = {}
for key, vals in confDict_obstrat.items():
    metDict[key] = eval('opts.{}'.format(key))


metricList = [SNObsStratPixel(**metDict)]
"""
fieldType = yaml_params['Observations']['fieldtype']
fieldName = yaml_params['Observations']['fieldname']
nside = yaml_params['Pixelisation']['nside']
saveData = 0
outDir = yaml_params['OutputSimu']['directory']
# now perform the processing
"""

# print('seasons and metric', opts.Observations_season,
#      metricList, opts.pixelmap_dir, opts.npixels)

procDict['fieldType'] = 'WFD'
procDict['metricList'] = metricList
procDict['fieldName'] = 'WFD'
# procDict['outDir'] = outDir
procDict['pixelList'] = opts.pixelList
procDict['nside'] = opts.nside

if __name__ == '__main__':
    toimport = 'from sn_tools.sn_process import Process'

    if opts.code == 'new':
        toimport = 'from sn_tools.sn_process_new import Process'

    exec(toimport)

    del procDict['code']

    # set the start method
    # multiprocessing.set_start_method('forkserver')
    process = Process(**procDict)
