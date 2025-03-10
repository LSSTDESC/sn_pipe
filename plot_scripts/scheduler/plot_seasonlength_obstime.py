#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:28:19 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_plotter_scheduler.plot_scheduler import add_info, ana_season
from sn_plotter_scheduler.plot_scheduler import plot_season_length, plot
import pandas as pd
import glob
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser(
    description='Script to plot DDF season length vs nvisits')
parser.add_option("--dirFiles", type=str, default='../sn_ddf_scheduler',
                  help="directory files [%default]")
parser.add_option("--plots", type=str, default='nvisits_season,season_length_nvisits',
                  help="plots [%default]")

opts, args = parser.parse_args()

theDir = opts.dirFiles
plots = opts.plots.split(',')

fis = glob.glob('{}/*.hdf5'.format(theDir))

df = pd.DataFrame()
for fi in fis:
    ddf = pd.read_hdf(fi)
    df = pd.concat((df, ddf))


df = add_info(df)
print(df.columns)

targets = df['target'].unique()

if 'nvisits_season' in plots:
    plot(df)

# estimate season length - take a central season, 5, for this estimation

if 'season_length_nvisits' in plots:
    idx = df['season'] == 5
    sel_df = df[idx]

    res_season = sel_df.groupby(['target']).apply(
        lambda x: ana_season(x)).reset_index()

    plot_season_length(res_season)

plt.show()
