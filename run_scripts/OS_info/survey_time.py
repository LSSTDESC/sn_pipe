#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:03:02 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_tools.sn_cadence_tools import Survey_depth, Survey_time
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
    Survey_time(dbDir, configFile)

if 'depth' in whats:
    Survey_depth(dbDir, configFile)
