#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:01:54 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import os
import numpy as np
from optparse import OptionParser

parser = OptionParser(
    description='Script to generate fake scenarios (csv files).')

parser.add_option('--budget_DD', type=float, default=0.07,
                  help='DD budget [%default]')
parser.add_option("--Nv_DD_max", type=int,
                  default=3500,
                  help="max number of Nvisits per DD/season [%default]")
parser.add_option("--Nf_combi", type=str,
                  default='\'(2,2),(2,3),(2,4)\'',
                  help="to show plot or not [%default]")
parser.add_option("--zcomp", type=str,
                  default='0.80,0.75,0.70',
                  help="redshift completeness configuration [%default]")
parser.add_option("--scen_names", type=str,
                  default='DDF_DESC_0.75, DDF_DESC_0.70,DDF_DESC_0.65',
                  help="scenario names corresponding to zcomp [%default]")
parser.add_option("--showPlot", type=int,
                  default=0,
                  help="to show plot or not [%default]")

opts, args = parser.parse_args()

params = vars(opts)

params['budget_DD'] = np.round(params['budget_DD'], 2)

# create strategies
script = 'run_scripts/desc_ddf_strategy/ddf_cohesive_strategy.py'
cmd_a = 'python {} --showPlot={}'.format(script, params['showPlot'])
for key, vals in params.items():
    cmd_a += ' --{}={}'.format(key, vals)

os.system(cmd_a)

# create scenarios
script = 'run_scripts/desc_ddf_strategy/build_scenarios_from_csv.py'
scenario = 'scenarios_{}.csv'.format(params['budget_DD'])
configscen = 'input/DESC_cohesive_strategy/config_scenarios.csv'
cmd_b = 'python {}'.format(script)
cmd_b += ' --configFile={}'.format(scenario)
cmd_b += ' --configScenario={}'.format(configscen)

os.system(cmd_b)

# generate dbs

cmd_c = 'python run_scripts/fakes/loop_scenarios.py --budget={}'.format(
    params['budget_DD'])

os.system(cmd_c)
