#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:40:16 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_io import checkDir
import pandas as pd
import numpy as np


def build_survey(survey_field_config, survey_field_visits):
    """
    Function to load the DDF survey config

    Parameters
    ----------
    survey_field_config : str
        Survey field config file.
    survey_field_visits : str
        survey config visits file

    Returns
    -------
    df : pandas df
        survey.

    """

    dfa = pd.read_csv(survey_field_config, comment='#')

    dfb = pd.read_csv(survey_field_visits, comment='#')

    df = dfa.merge(dfb, left_on=['fieldType'], right_on=[
                   'target'], suffixes=['', ''])

    df = df.drop(columns=['fieldType', 'target'])
    df = df.rename(columns={'fieldName': 'field'})

    # correct for EDFs
    df['field_str'] = df['field'].str.split('_', expand=True)[0]
    idx = df['field_str'] == 'EDFS'
    sel = pd.DataFrame(df[idx])
    sel['season_seq'] /= 2
    sel['season_seq'] = sel['season_seq'].astype(int)

    dfb = df[~idx]
    dfb = pd.concat((dfb, sel))

    dfb = dfb.drop(columns=['field_str'])
    return dfb


parser = OptionParser(
    description='Script to produce DDF input file to generate DDF tables for the LSST scheduler')

parser.add_option('--inputDir', type=str,
                  default='input/scheduler',
                  help='Location dir of input files [%default]')
parser.add_option('--outputDir', type=str,
                  default='../survey_lsst_scheduler',
                  help='output dir of the produced files [%default]')
parser.add_option('--ddf_survey_fields', type=str,
                  default='deep_rolling_survey.csv',
                  help='config file for fields [%default]')
parser.add_option('--ddf_survey_visits', type=str,
                  default='ddf_desc_0.70_sn.csv',
                  help='config file for visits [%default]')

opts, args = parser.parse_args()

inputDir = opts.inputDir
outputDir = opts.outputDir
ddf_survey_fields = opts.ddf_survey_fields
ddf_survey_visits = opts.ddf_survey_visits

# check if output dir exist
checkDir(outputDir)

# build the survey
df_survey = build_survey(f'{inputDir}/{ddf_survey_fields}',
                         f'{inputDir}/{ddf_survey_visits}')

# save outputfile
outName = '{}/{}.npy'.format(outputDir, ddf_survey_visits.split('.csv')[0])

np.save(outName, df_survey.to_records(index=False))
