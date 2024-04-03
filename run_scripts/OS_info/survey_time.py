#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:03:02 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from sn_tools.sn_obs import season
from sn_tools.sn_cadence_tools import Survey_depth, Survey_time
import pandas as pd
import operator
import os
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--dbDir", type=str,
                  default='../DB_Files',
                  help="data dir[%default]")
parser.add_option("--configFile", type=str,
                  default='input/cadence/config_ana_paper_plot.csv',
                  help="configuration file [%default]")
parser.add_option("--what", type=str,
                  default='survey_time,depth',
                  help="what to estimate [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
configFile = opts.configFile
whats = opts.what.split(',')


if 'survey_time' in whats:
    # estimate_survey_time(dbDir, configFile)
    Survey_time(dbDir, configFile)

if 'depth' in whats:
    # estimate_depth(dbDir, configFile)
    Survey_depth(dbDir, configFile)

"""
idx = df['season'] == 1
sela = df[idx]

print(sela)
"""
