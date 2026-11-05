#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:59:43 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_analysis.sn_os_atmos_calib_tools import load_OS,ddf_wfd
import numpy as np
from sn_tools.os_auxtel_tools import AuxTelSurvey
from sn_tools.sn_io import checkDir

parser = OptionParser(description='auxtel survey design')

confDict = make_dict_from_config('input_script', 'auxtel_survey_design.txt')

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params=vars(opts)

#outputdir
if params['outDir'] != 'None':
    checkDir(params['outDir'])

#load obs

obs = load_OS(params['dbDir'], params['dbName'], params['dbExtens'])

#add year

obs['year'] = obs['night']/366+1

obs['year'] = obs['year'].astype(int)

nside=32
ddf,wfd = ddf_wfd(obs.to_records(index=False),nside=nside)

#concatenate

obs = np.concat((ddf,wfd))

obs.sort(order='mjd')

print(obs.dtype)
year = params['year']

star_list=params['star_calib_file']
survey = AuxTelSurvey(obs,star_list,year=year,
                      survey_type=params['survey_type'],
                      nproc=params['nproc'])
df_survey = survey()

print(df_survey)
outName = '{}/auxtel_survey_{}_{}_{}.hdf5'.format(params['outDir'],
                                                  params['dbName'],
                                            params['survey_type'],year)
df_survey.to_hdf(outName,key='auxtel_survey')
"""
survey = AuxTelSurvey(obs,params['dbName'],
                      nside=nside,
                      figDir='../os_auxtel_survey',
                      #movieDir='../os_auxtel_movie',
                      year=year)

nights=range(1,9)
nproc = params['nproc']

nights = range(1,366)
nproc=30

nights = [1,2,3,4]
nproc=1
survey(nights,nproc)
"""