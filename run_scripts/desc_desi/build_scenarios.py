#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:27:46 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import os


def get_survey(df, scenType):
    """
    Function to grab the survey

    Parameters
    ----------
    df : pandas df
        Data to process.
    scenType : str
        scenario type (desi,desi2,4hs,crs).

    Returns
    -------
    pandas df
        selected scenario.

    """

    idx = df['scenType'] == scenType

    return df[idx]


def loopIt(iscen, surv_dict, dfa, dfb=None, dfc=None):
    """
    Function to combine scenarios

    Parameters
    ----------
    iscen : int
        scenario number.
    surv_dict : dict
        dict of scenarios.
    dfa : pandas df
        Data to consider.
    dfb : pandas df, optional
        Data to consider. The default is None.
    dfc : pandas df, optional
        Data to consider. The default is None.

    Returns
    -------
    iscen : int
        last scenario number.
    surv_dict : dict
        dict of scenarios.

    """

    for i, row in dfa.iterrows():
        if dfb is None:
            iscen += 1
            surv_dict['scen_{}'.format(iscen)] = row['survey']
        else:
            for j, rowb in dfb.iterrows():
                if dfc is None:
                    iscen += 1
                    surv = '{}/{}'.format(row['survey'], rowb['survey'])
                    surv_dict['scen_{}'.format(iscen)] = surv
                else:
                    for k, rowc in dfc.iterrows():
                        iscen += 1
                        surv = '{}/{}/{}'.format(row['survey'],
                                                 rowb['survey'],
                                                 rowc['survey'])
                        surv_dict['scen_{}'.format(iscen)] = surv

    return iscen, surv_dict


def get_full_survey(df):
    """
    Function to get a full survey

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    dfb : pandas df
        output data.

    """

    surveys = df['survey'].unique()
    r = ['/'.join(surveys)]

    dfb = pd.DataFrame(r, columns=['survey'])

    return dfb


parser = OptionParser('script to build survey scenarios for desi/desi2')

parser.add_option("--lookup_desi", type=str,
                  default='lookup_desi.csv',
                  help="DESI/DESI2 scenarios [%default]")
parser.add_option("--survey_ref", type=str,
                  default='survey_scenario_ref.csv',
                  help="reference scenario to append [%default]")
parser.add_option("--outDir", type=str,
                  default='desc_desi_surveys',
                  help="output dir for scenarios [%default]")

opts, args = parser.parse_args()

lookup_desi = opts.lookup_desi
survey_ref = opts.survey_ref
outDir = opts.outDir

# load desi lookup table (survey def.)
scen_desi = pd.read_csv(lookup_desi, comment='#')

print(scen_desi['survey'])

scentypes = scen_desi['scenType'].unique()

# load surveys

desi_surveys = get_survey(scen_desi, 'desi')
desi2_survey = get_survey(scen_desi, 'desi2')
crs_survey = get_survey(scen_desi, 'crs')
hs_survey = get_survey(scen_desi, '4hs')
surv_dict = {}

desi_survey = get_full_survey(desi_surveys)

iscen = 0
# make surveys
iscen, surv_dict = loopIt(iscen, surv_dict, scen_desi)
for vv in [desi_surveys, desi_survey]:
    iscen, surv_dict = loopIt(iscen, surv_dict, vv, desi2_survey)
    iscen, surv_dict = loopIt(iscen, surv_dict, vv, crs_survey)
    iscen, surv_dict = loopIt(iscen, surv_dict, vv, hs_survey)


print(surv_dict)

# create new files
field = 'WFD'
fieldType = 'WFD'
nsn_max_season = '1.e4'
zType = 'spectroz_nosat'
zmax = 0.6

for key, vals in surv_dict.items():
    new_file = '{}/survey_scenario_{}.csv'.format(outDir, key)
    cmd = 'scp {} {}'.format(survey_ref, new_file)
    os.system(cmd)
    to_app = vals.split('/')
    r = []
    for vv in to_app:
        idx = scen_desi['survey'] == vv
        sel_scen = scen_desi[idx]
        sur = vv
        season_min = sel_scen['season_min'].values[0]
        season_max = sel_scen['season_max'].values[0]
        host_effi = sel_scen['host_effi'].values[0]
        footprint = sel_scen['footprint'].values[0]
        ll = '{},{},{},{},{},{},{},{},{}_WFD,{}'.format(sur, field,
                                                        fieldType,
                                                        season_min, season_max,
                                                        nsn_max_season,
                                                        host_effi, zType,
                                                        footprint, zmax)
        r.append(ll)
    with open(new_file, 'a') as fd:
        for vv in r:
            fd.write('{} \n'.format(vv))
    # break
