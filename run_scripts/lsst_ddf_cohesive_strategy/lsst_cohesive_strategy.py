#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 11:03:48 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_io import checkDir


def get_ud_config(zcomp_file, cad_ud, sl_ud, nu_season=360, fieldtype='ud'):
    """
    Function to build UD fields

    Parameters
    ----------
    zcomp_file : str
        zcomp fine name.
    cad_ud : float
        UD cadence.
    sl_ud : float
        UD season length.
    nu_season : int, optional
        number of u-visits per season. The default is 360.
    fieldtype : str, optional
        Field type. The default is 'ud'.

    Returns
    -------
    ud_config : pandas df
        Resulting df.

    """

    nvisits_zcomp = pd.read_csv(zcomp_file, comment='#')

    # print(nvisits_zcomp)

    # the number of visits in nvisits_zcomp is of 1 night-> have to be modified

    ud_config = pd.DataFrame(nvisits_zcomp)

    ud_config['cad'] = cad_ud
    ud_config['sl'] = sl_ud

    bands = list('grizy')
    for b in bands:
        ud_config[b] *= ud_config['cad']

    ud_config['u'] = nu_season*ud_config['cad']/ud_config['sl']
    ud_config['u'] = np.ceil(ud_config['u'])
    ud_config['u'] = ud_config['u'].astype(int)
    ud_config['nvisits'] = ud_config[bands].sum(axis=1)

    ud_config = ud_config.rename(columns={'nvisits': 'nvisits_night'})
    ud_config['fieldtype'] = fieldtype

    return ud_config


def get_df_config(visits, cad_df, sl_df, zcomp=[]):
    """
    Function to build df fields

    Parameters
    ----------
    visits : pandas df
        Number of visits per season/band.
    cad_df : float
        df cadence.
    sl_df : float
        df season length.
    zcomp : list(float), optional
        zcomp values. The default is [].

    Returns
    -------
    df_config : pandas df
        Resulting df.

    """

    df_config = pd.DataFrame(visits)

    df_config['cad'] = cad_df
    df_config['sl'] = sl_df

    # print(df_config)

    # correct for the number of visits since they are given per season

    bands = list('ugrizy')
    for b in bands:
        df_config[b] *= df_config['cad']/df_config['sl']
        # df_config[b] = np.ceil(df_config[b])
        # df_config[b] = df_config[b].astype(int)
        df_config = df_config.round({b: 1})

    df_config['nvisits_night'] = df_config[bands].sum(axis=1)
    df_config = df_config.round({'nvisits_night': 1})

    df_zcomp = pd.DataFrame(zcomp, columns=['zcomp'])

    df_config = df_config.merge(df_zcomp, how='cross')

    return df_config


def get_nvisits(grp):
    """
    Function to estimate the number of visits per season

    Parameters
    ----------
    grp : pandas df
        Data to use for the estimation of Nvisits.

    Returns
    -------
    df : pandas df
        original data plus nvisits_season column.

    """

    grp['nvisits_season'] = grp['nvisits_night']*grp['sl']/grp['cad']

    res = grp['nvisits_season'].sum()

    df = pd.DataFrame([res], columns=['nvisits'])

    return df


def get_df(fName, cad_df, sl_df, zcomp):
    """
    Function to build df field

    Parameters
    ----------
    fName : str
        config file name (csv).
    cad_df : float
        df cadence.
    sl_df : float
        df season length.
    zcomp : list(float)
        zcomp values.

    Returns
    -------
    res : pandas df
        output df.

    """

    dd = pd.read_csv(fName, comment='#')
    res = get_df_config(dd, cad_df, sl_df, zcomp=zcomp)

    return res


def complete_df(grp, req):
    """
    Function to estimate the filter sequences for df fields

    Parameters
    ----------
    grp : pandas df
        Data to process.
    req : pandas df
        Requirements.

    Returns
    -------
    res : pandas df
        output data.

    """

    bands = list('ugrizy')

    # grab the total number of visits for each band/field
    for b in bands:
        grp['{}_season'.format(b)] = grp[b]*grp['sl']/grp['cad']

    bbands = list(map(lambda x:  x+'_season', bands))
    rr = grp.groupby(['field'])[bbands].sum().reset_index()
    for b in bands:
        rr = rr.rename(columns={'{}_season'.format(b): '{}_obs'.format(b)})

    # get the requirements (10 years) corresponding to this zcomp
    idx = req['zcomp'] == float(grp.name)
    idx &= req['fieldtype'] == 'df'
    sel_req = req[idx]
    print(sel_req)
    for b in bands:
        sel_req['{}_survey'.format(b)] = 9*sel_req[b] * \
            sel_req['sl']/sel_req['cad']

    # grab the field list
    fields = pd.DataFrame(grp['field'].unique(), columns=['field'])

    # merge fields with the requirements
    sel_req = sel_req.merge(fields, how='cross')
    bbou = list(map(lambda x:  x+'_survey', bands))
    bbou += ['field']
    print(sel_req[bbou])

    # merge with the original df
    rr = rr.merge(sel_req[bbou], left_on=['field'],
                  right_on=['field'], suffixes=['', ''])

    # estimate the remaining number of visits to get
    for b in bands:
        rr['delta_{}'.format(b)] = rr['{}_survey'.format(b)
                                      ]-rr['{}_obs'.format(b)]

    # if this number is negative, set it to 0!
    bboud = list(map(lambda x:  'delta_'+x, bands))

    pp = rr[bboud]
    idx = pp < 0
    rrr = rr[bboud+['field']]
    rr[idx] = 0

    print(rr)

    # complete the df fields with these infos to complete the survey
    idx = grp['fieldtype'] == 'df'

    ddf_season = grp[idx].groupby(
        ['field']).size().to_frame('nseasons').reset_index()

    rr = rr.merge(ddf_season, left_on=['field'], right_on=[
                  'field'], suffixes=['', ''])

    for b in bands:
        rr['{}_season'.format(b)] = rr['delta_{}'.format(b)]/rr['nseasons']

    bbb = list(map(lambda x:  x+'_season', bands))

    # rr['nvisits'] = rr[bbb].sum(axis=1)

    thecols = ['field']+bbb
    sel = grp[idx]
    sel = sel.merge(rr[thecols], left_on=['field'],
                    right_on=['field'], suffixes=['', '_y'])

    # finally: merging and cleaning
    for b in bands:
        thecol = '{}_season'.format(b)
        sel = sel.drop(columns=[thecol])
        sel = sel.rename(columns={'{}_y'.format(thecol): thecol})
        sel['{}'.format(b)] = sel[thecol]*sel['cad']/sel['sl']
        sel = sel.round({thecol: 1, b: 1})

    sel['nvisits_night'] = sel[bands].sum(axis=1)
    sel = sel.round({'nvisits_night': 1})

    res = pd.DataFrame(grp[~idx])
    res = pd.concat((res, sel))

    return res


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
parser.add_option("--ddf_scenario", type=str,
                  default='input/lsst_ddf_cohesive_strategy/ddf_survey_rolling_2.csv',
                  help="DDF scenario [%default]")
"""
parser.add_option("--nvisits_LSST", type=float,
                  default=2.e6,
                  help="Total number of LSST visits (10 years) [%default]")
"""
parser.add_option("--nvisits_req", type=str,
                  default='input/lsst_ddf_cohesive_strategy/nvisits_pz_wl_agn.csv',
                  help="nvisits req. by PZ+WL+AGN [%default]")
parser.add_option("--nvisits_req_y1", type=str,
                  default='input/lsst_ddf_cohesive_strategy/nvisits_pz_wl_agn_y1.csv',
                  help="nvisits req. by PZ+WL+AGN - y1[%default]")
parser.add_option("--survey_type", type=str,
                  default='realistic',
                  help="survey_type[%default]")
parser.add_option("--survey_output_name", type=str,
                  default='ddf_survey_rolling_2_complete.csv',
                  help="survey output name [%default]")
parser.add_option("--outDir", type=str,
                  default='../lsst_ddf_cohesive',
                  help="output directory [%default]")

opts, args = parser.parse_args()

zcomp_file = opts.zcomp_file
cad_ud = opts.cad_ud
sl_ud = opts.sl_ud
cad_df = opts.cad_df
sl_df = opts.sl_df
ddf_scenario = opts.ddf_scenario
nvisits_lsst = opts.nvisits_LSST
nvisits_req = opts.nvisits_req
nvisits_req_y1 = opts.nvisits_req_y1
survey_type = opts.survey_type
survey_output_name = opts.survey_output_name
outDir = opts.outDir

# all_ud fields
field_config = get_ud_config(zcomp_file, cad_ud, sl_ud)

# df
zcomp = field_config['zcomp'].to_list()
req_df = get_df(nvisits_req, cad_df, sl_df, zcomp)
req_df_y1 = get_df(nvisits_req_y1, cad_df, sl_df, zcomp)

if survey_type == 'realistic':
    field_config = pd.concat((field_config, req_df))

if survey_type == 'science_fiction':
    field_config = pd.concat((field_config, req_df_y1))

# load the ddf scenario

ddf_scen = pd.read_csv(ddf_scenario, comment='#')


# construction of the survey for all zcomp

ddf_survey = ddf_scen.merge(field_config, left_on=['fieldtype'], right_on=[
                            'fieldtype'], suffixes=['', ''])


if survey_type == 'science_fiction':
    ddf_survey = ddf_survey.groupby(['zcomp']).apply(
        lambda x: complete_df(x, req_df))


idx = ddf_survey['zcomp'] <= 0.8
ddf_survey = ddf_survey[idx]

# res = res.sort_values(by=['zcomp', 'field', 'year'])

fName = '{}/{}'.format(outDir, survey_output_name)
ddf_survey.to_csv(survey_output_name)

"""
# now process this data
ddf_survey = ddf_survey.set_index('zcomp')
res = ddf_survey.groupby(['zcomp']).apply(
    lambda x: get_nvisits(x)).reset_index()
res['budget'] = res['nvisits']/nvisits_lsst


fig, ax = plt.subplots()

ax.plot(res['zcomp'], res['budget'])
ax.grid(visible=True)
plt.show()
print(res)
"""
