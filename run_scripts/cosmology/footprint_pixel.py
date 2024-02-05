#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:56:31 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import healpy as hp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_map(nside) -> pd.DataFrame:
    """
    Function to get a pixel map

    Parameters
    ----------
    nside : int
        nside healpix parameter.

    Returns
    -------
    map_pixel : pandas df
        cols: pix_RA, pix_Dec, weight, healpixID.

    """

    # get the total number of pixels
    npixels = hp.nside2npix(nside)

    # pixels = hp.get_all_neighbours(nside, 0.0, 0.0, nest=True, lonlat=Tru)

    # get the (RA, Dec) of the pixel centers
    vec = hp.pix2ang(nside, range(npixels), nest=True, lonlat=True)

    map_pixel = pd.DataFrame(range(npixels), columns=['healpixID'])
    map_pixel['pix_RA'] = vec[0]
    map_pixel['pix_Dec'] = vec[1]
    map_pixel['weight'] = 1

    return map_pixel


def plot_pixels(data):
    """
    Function to plot pixels with weight >= 0

    Parameters
    ----------
    data : pandas df
        data to plot.

    Returns
    -------
    None.

    """

    npixels = len(data)
    hpxmap = np.zeros(npixels, dtype=np.float)
    hpxmap = np.full(hpxmap.shape, 0.)
    hpxmap[data['healpixID']] += data['weight']

    xmin = 0.0
    xmax = np.max(data['weight'])
    norm = plt.cm.colors.Normalize(xmin, xmax)

    cmap = plt.cm.jet
    cmap.set_under('w')

    hp.mollview(hpxmap, cmap=cmap, nest=True,
                min=xmin, max=xmax, norm=norm,
                title='E(B-V) MW - SFD')
    hp.graticule()


parser = OptionParser()
parser.add_option("--nside", type=int, default=64,
                  help="nside healpix parameter[%default]")

opts, args = parser.parse_args()

nside = opts.nside

map_pixel = get_map(nside)

print(map_pixel)

plot_pixels(map_pixel)

# flag dec < - 67 deg
idx = map_pixel['pix_Dec'] < -67.0
map_pixel.loc[idx, 'weight'] = -1
plot_pixels(map_pixel)

# exclude MW zone
xc = 0.8*360/24
yc = 17.14
radius_1 = 53.43+yc
radius_2 = 66.87+yc

idx = map_pixel['pix_RA'] >= 180.
map_pixel.loc[idx, 'pix_RA'] -= 360.
map_pixel['dist'] = (map_pixel['pix_RA']-xc)**2+(map_pixel['pix_Dec']-yc)**2

print(map_pixel[['pix_RA', 'pix_Dec', 'dist']])
idxa = map_pixel['dist'] < radius_1**2
sela = map_pixel[idxa]
idxb = map_pixel['dist'] < radius_2**2
selb = map_pixel[idxb]
idxc = selb['healpixID'].isin(sela['healpixID'].to_list())
ll = selb.loc[~idxc, 'healpixID'].to_list()

idx = map_pixel['healpixID'].isin(ll)
map_pixel.loc[idx, 'weight'] = -1
plot_pixels(map_pixel)

plt.show()
