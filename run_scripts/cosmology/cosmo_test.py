#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:32:52 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import os


def process(dataDir_DD, dbName_DD, dataDir_WFD,
            dbName_WFD, timescale, survey_scenario, tagsurvey):
    """
    Function to launch cosmology estimation

    Parameters
    ----------
    dataDir_DD : str
        Dir for DD data.
    dbName_DD : str
        dbName for DD.
    dataDir_WFD : str
        Dir for WFD data.
    dbName_WFD : str
        dbName for WFD.
    timescale : str
        Time scale (year/season).
    survey_scenario : str
        survey csv file.
    tagsurvey : str
        tag for the survey name.

    Returns
    -------
    None.

    """

    cmd = 'python run_scripts/cosmology/cosmology.py'
    cmd += ' --dataDir_DD={}'.format(dataDir_DD)
    cmd += ' --dbName_DD={}'.format(dbName_DD)
    cmd += ' --dataDir_WFD={}'.format(dataDir_WFD)
    cmd += ' --dbName_WFD={}'.format(dbName_WFD)
    cmd += ' --timescale={}'.format(timescale)
    cmd += ' --survey={}'.format(survey_scenario)
    cmd += ' --tagsurvey={}'.format(tagsurvey)
    cmd += ' --outDir=../cosmo_fit_WFD_sigmaC_test'
    cmd += ' --seasons=1,2,3,4,5,6,7,8,9,10'
    cmd += ' --surveyDir=../test_survey'
    cmd += ' --nrandom=1'
    cmd += ' --plot_test=0'
    cmd += ' --test_mode=0'
    cmd += ' --nproc=1'
    cmd += ' --low_z_opti=0'

    os.system(cmd)


parser = OptionParser()
parser.add_option("--dataDir_DD", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_z_smflux_test_G10_JLA',
                  help="dir for DD data [%default]")
parser.add_option("--dbName_DD", type=str,
                  default='baseline_v3.0_10yrs',
                  help="dbName for DD survey[%default]")
parser.add_option("--dataDir_WFD", type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_test_G10_JLA',
                  help="dir for WFD data [%default]")
parser.add_option("--dbName_WFD", type=str,
                  default='baseline_v3.0_10yrs',
                  help="dbName for WFD survey[%default]")
parser.add_option("--timescale", type=str,
                  default='year',
                  help="time scale for the processing (year/season) [%default]")
parser.add_option("--survey_scenario", type=str,
                  default='survey_scenario_ti.csv',
                  help="survey scenario [%default]")
parser.add_option("--tagsurvey", type=str,
                  default='ti',
                  help="survey name tag[%default]")

opts, args = parser.parse_args()

process(**vars(opts))
