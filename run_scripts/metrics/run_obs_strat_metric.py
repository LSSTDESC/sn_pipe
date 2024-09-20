#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:50:42 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import sn_script_input
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_io import add_parser
from optparse import OptionParser

# get all possible simulation parameters and put in a dict
path_process_input = sn_script_input.__path__

confDict_gen = make_dict_from_config(
    path_process_input[0], 'config_process.txt')

parser = OptionParser()

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict_gen)

opts, args = parser.parse_args()

procDict = {}
for key, vals in confDict_gen.items():
    procDict[key] = eval('opts.{}'.format(key))

if __name__ == '__main__':
    toimport = 'from sn_tools.sn_process import Process'

    if opts.code == 'new':
        toimport = 'from sn_tools.sn_process_new import Process'

    exec(toimport)

    del procDict['code']

    # set the start method
    # multiprocessing.set_start_method('forkserver')
    process = Process(**procDict)
