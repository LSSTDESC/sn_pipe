#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 11:03:48 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd


def get_ud_config(zcomp_file, cad_ud, sl_ud):
    
    nvisits_zcomp = pd.read_csv(zcomp_file,comment='#')
    
    print(nvisits_zcomp)
    
    #the number of visits in nvisits_zcomp is of 1 night-> have to be modified 
    
    ud_config = pd.DataFrame(nvisits_zcomp)
    
    ud_config['cad'] = cad_ud
    ud_config['sl'] = sl_ud    

    bands = list('grizy')
    ud_config[bands] = ud_config[bands]*cad_ud
    
    ud_config['nvisits'] = ud_config[bands].sum(axis=1)
    ud_config['u'] = 0
    
    return ud_config

def get_df_config(visits,cad_df,sl_df):
    
    
    df_config = pd.DataFrame.from_dict(visits)
    
    df_config['cad'] = cad_df
    df_config['sl'] = sl_df
    
    print(df_config)
    
    # correct for the number of visits since they are given per season
    
    bands = list('ugrizy')
    df_config[bands] = df_config[bands]*df_config['sl']/df_config['cad']
    
    df_config['nvisits_night'] = df_config[bands].sum(axis=1)
    
    return df_config
    

parser = OptionParser(description='Design a cohesive LSST DDF Strategy')

parser.add_option("--zcomp_file", type=str,
                  default='input/DESC_cohesive_strategy/Nvisits_zcomp_paper.csv',
                  help="input file for SNe Ia depth[%default]")
parser.add_option("--cad_ud", type=int,
                  default=2,
                  help="UD cadence [%default]")
parser.add_option("--sl_ud", type=int,
                  default=210,
                  help="UD season length [%default]")
parser.add_option("--cad_df", type=int,
                  default=2,
                  help="DF cadence [%default]")
parser.add_option("--sl_df", type=int,
                  default=180,
                  help="DF season length [%default]")


opts, args = parser.parse_args()

zcomp_file = opts.zcomp_file
cad_ud = opts.cad_ud
sl_ud = opts.sl_ud
cad_df = opts.cad_df
sl_df = opts.sl_df

# configuration field dict
field = {}

#ud fields
field['ud'] = get_ud_config(zcomp_file, cad_ud, sl_ud)

# df
bands = 'ugrizy'
nv_y1 = [360,165,184,270,450,360]
nv_df = [360,139,212,288,450,360]
nv_y1 = map(lambda x:[x], nv_y1)
nv_df = map(lambda x:[x], nv_df)

visits_y1=dict(zip(bands,nv_y1))
visits_df = dict(zip(bands,nv_df))

field['df_y1'] = get_df_config(visits_y1, cad_df, sl_df)
field['df'] = get_df_config(visits_df, cad_df, sl_df)
print(field)