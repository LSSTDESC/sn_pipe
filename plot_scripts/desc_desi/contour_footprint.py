#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:02:07 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import numpy as np
import healpy as hp


class Contours:
    def __init__(self, x, dx, y, dy):

        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

        self.dfpix = self.get_dfpix

    def get_dfpix(self):

        dfpix_x = self.fill_pix(self.x, self.dx,
                                col_index='i', col_min='xmin', col_max='xmax')
        dfpix_y = self.fill_pix(self.y, self.dy,
                                col_index='j', col_min='ymin', col_max='ymax')

        print(dfpix_x)
        print(dfpix_y)
        dfpix = dfpix_x.merge(dfpix_y, how='cross')
        return dfpix

    def fill_pix(self, x, dx, col_index='i', col_min='xmin', col_max='xmax'):

        dfpix = pd.DataFrame(x, columns=[col_min])
        dfpix[col_index] = dfpix.index
        dfpix[col_max] = dfpix[col_min]+dx

        return dfpix

    def __call__(self, df):

        self.process(df)

    def process(self, data):

        N = int(len(data)/50)

        vv = np.split(data, N)

    def process_single(self, data):

        dfb = self.dfpix.merge(data[:50], how='cross')

        idx = dfb['pixRA'] >= dfb['xmin']
        idx &= dfb['pixRA'] < dfb['xmax']
        idx &= dfb['pixDec'] >= dfb['ymin']
        idx &= dfb['pixDec'] < dfb['ymax']

        res = dfb[idx][['i', 'j']]

        return pd.DataFrame(res)


def pix_coord(df, nside=64):

    healpixId = df['healpixID'].to_list()
    coord = hp.pix2ang(nside, healpixId, nest=True, lonlat=True)

    pixRA = coord[0]
    pixDec = coord[1]
    df['pixRA'] = pixRA
    df['pixDec'] = pixDec

    return df


def path(inputDir, sname):

    pp = '{}/footprint_{}_WFD.hdf5'.format(inputDir, sname)

    return pp


dx = 0.1
dy = 0.1
x = np.arange(0., 360., dx)
y = np.arange(-90., 90., dy)

cont = Contours(x, dx, y, dy)

inputDir = 'input/cosmology/footprints'
footprint = 'TiDES'

data = pd.read_hdf(path(inputDir, footprint))
cont(pix_coord(data)[['pixRA', 'pixDec']])
