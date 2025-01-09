#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:54:16 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sn_tools.sn_visu import get_map, plot_pixels


def put_in_map(map_pixel, df, icolor):

    ll = df['healpixID'].to_list()
    idx = map_pixel['healpixID'].isin(ll)
    map_pixel.loc[idx, 'weight'] = icolor

    return map_pixel


def path(inputDir, sname):

    pp = '{}/footprint_{}_WFD.hdf5'.format(inputDir, sname)

    return pp


inputDirf = 'input/cosmology/footprints'
inputDirc = 'input/cosmology/contours'

footprints = ['TiDES', 'desi2_v3', 'crs_v3', 'desi_v3', '4hs_v3']

footprints = ['TiDES', 'desi_v3', '4hs_v3', 'desi2_v3', 'crs_v3']
footprints = ['TiDES', 'desi_v3', '4hs_v3', 'desiext_v3', 'crs_v3','desi2_v3']
inDir = [inputDirf,inputDirf,inputDirc,inputDirc,inputDirc,inputDirc]

dict_dir = dict(zip(footprints,inDir))
nside = 64
map_pixel = get_map(nside)

for i, footp in enumerate(footprints):
    pp = path(dict_dir[footp], footp)
    df = pd.read_hdf(pp)
    print(df.columns)
    map_pixel = put_in_map(map_pixel, df, i+2)


plot_pixels(map_pixel, xticklabels=footprints,imax=6)

plt.show()
