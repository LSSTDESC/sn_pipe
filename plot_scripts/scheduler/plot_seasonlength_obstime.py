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

theDir = '../ddf_scheduler'

fis = glob.glob('{}/*.hdf5'.format(theDir))

df = pd.DataFrame()
for fi in fis:
    ddf = pd.read_hdf(fi)
    df = pd.concat((df, ddf))


df = add_info(df)
print(df.columns)

targets = df['target'].unique()

plot(df)

# estimate season length - take a central season, 5, for this estimation

idx = df['season'] == 5
sel_df = df[idx]

print('bobobobo', sel_df['target'].unique())

res_season = sel_df.groupby(['target']).apply(
    lambda x: ana_season(x)).reset_index()

print(res_season)

plot_season_length(res_season)

plt.show()
