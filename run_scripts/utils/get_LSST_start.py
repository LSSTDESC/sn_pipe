#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:48:53 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_obs import getObservations
import pandas as pd
import numpy as np

parser = OptionParser(description='Script to estimate LSST survey mjd start')

parser.add_option("--dbList", type=str,
                  default='dbList.csv',
                  help="db list to process[%default]")
parser.add_option("--outName", type=str,
                  default='LSSTStart.csv',
                  help="output file name[%default]")

opts, args = parser.parse_args()

dbList = opts.dbList
outName = opts.outName
# dblist csv file should contain at least the following fields:
# dbDir,dbExtens,dbName

ll = pd.read_csv(dbList, comment='#')

r = []
for i, row in ll.iterrows():
    dbName = row['dbName']
    obs = getObservations(row['dbDir'], dbName, row['dbExtens'])

    r.append((dbName, np.min(obs['mjd'])))

df = pd.DataFrame(r, columns=['dbName', 'LSSTStart'])

df.to_csv(outName, index=False)
