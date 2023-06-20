#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:01:54 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import os

# create strategies
cmd_a = 'python \
    run_scripts/desc_ddf_strategy/ddf_cohesive_strategy.py --showPlot=1'
os.system(cmd_a)

# create scenarios
cmd_b = 'python run_scripts/desc_ddf_strategy/build_scenarios_from_csv.py \
    --configScenario=input/DESC_cohesive_strategy/config_scenarios.csv'

os.system(cmd_b)

# generate dbs

cmd_c = 'python run_scripts/fakes/loop_scenarios.py'

os.system(cmd_c)
