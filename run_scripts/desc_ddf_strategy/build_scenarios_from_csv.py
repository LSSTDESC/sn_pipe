#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:50:14 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser


def modif_UD(df, cad):
    """
    Function to modify UD fields cadence when they behave like DD

    Parameters
    ----------
    df : pandas df
        data to process.
    cad : float
        cadence of observation.

    Returns
    -------
    res : pandas df
        result.

    """

    # consider UD fields only
    idx = df['fieldType'] == 'UD'

    df_res = df[~idx]

    sel = df[idx]

    fields = sel['field'].unique()

    bands = 'ugrizy'
    cols = []
    for b in bands:
        cols.append('Nvisits_{}'.format(b))

    res = pd.DataFrame()
    for nn in fields:
        idxb = sel['field'] == nn
        selb = sel[idxb]

        fieldType = selb['fieldType'].unique()
        if fieldType[0] == 'UD':
            selb = modif_UD_indiv(selb, cols, cad)

        res = pd.concat((res, selb))

    res = pd.concat((res, df_res))

    return res


def modif_UD_indiv(selb, cols, cad):
    """
    Function to modify UD fields cadence when they behave like DD

    Parameters
    ----------
    selb : pandas df
        data to process.
    cols : list(str)
        columns used to make the sum.
    cad : float
        cadence of ob..

    Returns
    -------
    seld : pandas df
        resulting df.

    """

    # estimate the total number of visits
    selb['Nvisits'] = selb[cols].sum(axis=1)
    selb['Nvisits'] = selb['Nvisits'].astype(int)

    # get the max number of visits
    max_visits = selb['Nvisits'].max()

    # change the cadence of the row with the lowest number of visits

    idxc = selb['Nvisits'] < max_visits

    selc = selb[idxc]
    seld = selb[~idxc]

    for b in bands:
        selc['cadence_{}'.format(b)] = cad

    seld = pd.concat((seld, selc))

    seld = seld.drop(columns=['Nvisits'])

    return seld


def modif_Euclid(df, fields=['DD:EDFS_a', 'DD:EDFS_b']):
    """
    Function to modify cadences of Euclid fields

    Parameters
    ----------
    df : pandas df
        data to process.
    fields : list(str), optional
        List of fields to modify. The default is ['DD:EDFS_a', 'DD:EDFS_b'].

    Returns
    -------
    dftot : pandas df
        result.

    """

    idx = df['field'].isin(fields)

    dftot = df[~idx]

    sel = df[idx]
    # double the cadence for these fields
    bands = 'ugrizy'

    for b in bands:
        sel['cadence_{}'.format(b)] *= 2

    print(sel)

    dftot = pd.concat((dftot, sel))

    return dftot


parser = OptionParser(
    description='Script to generate fake scenarios (csv files).')

parser.add_option('--configFile', type='str', default='scenarios.csv',
                  help='config file to use[%default]')
parser.add_option('--outputDir', type='str',
                  default='input/DESC_cohesive_strategy',
                  help='output dir[%default]')

opts, args = parser.parse_args()

configFile = opts.configFile

# load scenarios

df_scen = pd.read_csv(configFile)

print(df_scen['name'].unique())

scenarios = df_scen['name'].unique()

DD_names = ['DD:COSMOS', 'DD:XMM_LSS', 'DD:ECDFS',
            'DD:ELAISS1', 'DD:EDFS_a', 'DD:EDFS_b']
DD_types = ['UD', 'UD', 'DD', 'DD', 'DD', 'DD']
cad = [2, 2, 3, 3, 3, 3, 3]
sl = [180]*len(DD_names)

bands = 'ugrizy'

for scen in scenarios:
    idx = df_scen['name'] == scen
    sel_scen = df_scen[idx]
    dfa_scen = pd.DataFrame()
    for i, ddn in enumerate(DD_names):
        ido = sel_scen['fieldType'] == DD_types[i]
        dfa = pd.DataFrame(sel_scen[ido])
        for b in bands:
            dfa['cadence_{}'.format(b)] = cad[i]
            dfa = dfa.rename(columns={'{}'.format(b): 'Nvisits_{}'.format(b)})
        dfa = dfa.rename(columns={'year': 'seasons'})
        dfa['seasonLength'] = sl[i]
        dfa['field'] = DD_names[i]
        dfa_scen = pd.concat((dfa_scen, dfa))

    dfa_scen = modif_UD(dfa_scen, cad[3])
    dfa_scen = modif_Euclid(dfa_scen)
    print(dfa_scen)
    outName = '{}/{}.csv'.format(opts.outputDir, scen)
    dfa_scen.to_csv(outName, index=False)
    # break
