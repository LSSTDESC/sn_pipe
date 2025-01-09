#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:02:07 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import numpy as np
import healpy as hp
import time
from sn_tools.sn_utils import multiproc
from sn_tools.sn_visu import pix_coord,get_all_pixels
from skimage import measure


class Contours:
    def __init__(self, x, dx, y, dy):
        """
        Tentative class to project pixels

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        dx : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        dy : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

        self.dfpix = self.get_dfpix()

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

        params = {}

        bb = multiproc(df, params, self.process, nproc=8)

        return bb

    def process(self, data, params, j=0, output_q=None):

        N = int(len(data)/100)

        vv = np.array_split(data, N)


        res = pd.DataFrame()
        for tab in vv:
            rr = self.process_single(tab, j)
            res = pd.concat((res, rr))

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def process_single(self, data, j):

        dfb = self.dfpix.merge(data, how='cross')

        idx = dfb['pixRA'] >= dfb['xmin']
        idx &= dfb['pixRA'] < dfb['xmax']
        idx &= dfb['pixDec'] >= dfb['ymin']
        idx &= dfb['pixDec'] < dfb['ymax']

        res = dfb[idx][['i', 'j']]

        res = res.drop_duplicates()

        return pd.DataFrame(res)

def path(inputDir, sname):
    """
    Function to grab the full path of a file

    Parameters
    ----------
    inputDir : str
        Input dir.
    sname : str
        survey name.

    Returns
    -------
    pp : str
        Full path.

    """

    pp = '{}/footprint_{}_WFD.hdf5'.format(inputDir, sname)

    return pp


def sel_pixels(df, valmin, valmax, col):
    """
    
    Function to select pixels
    
    Parameters
    ----------
    df : pandas df
        Data to process.
    valmin : float
        min sel val.
    valmax : float
        max sel val.
    col : str
        column.

    Returns
    -------
    pandas df
        filtered df.

    """
    

    idx = df[col] >= valmin
    idx &= df[col] < valmax

    return df[idx]


def match_pixels(pixref, data, ra_min=10., delta_ra=0.1):
    """
    Function to match pixels from the celestial sphere

    Parameters
    ----------
    pixref : pandas df
        pixel of the celestial sphere.
    data : pandas df
        Data to process.
    ra_min : float, optional
        min ra. The default is 10..
    delta_ra : float, optional
        Delta_ra. The default is 0.1.

    Returns
    -------
    ll : list(int)
        List of pixels

    """

    ra_max = ra_min+delta_ra

    # select all_pixels
    pixsel = sel_pixels(pixref, ra_min, ra_max, 'pixRA')
    datasel = sel_pixels(data, ra_min, ra_max, 'pixRA')

    pixsel = pixsel.sort_values(by=['pixDec'])

    pixlist = datasel['healpixID'].to_list()

    inside = False
    previous_in = False
    ll = []

    for i, row in pixsel.iterrows():
        hpixID = row['healpixID']
        if hpixID in pixlist:
            if not previous_in:
                ll.append(hpixID)
                inside = True
            previous_in = True
            hpixID_p = hpixID
        else:
            if previous_in:
                ll.append(hpixID_p)
            previous_in = False

    return ll

def get_contours(all_pixels,data,footprint,outName):
    """
    Functio to get contours

    Parameters
    ----------
    all_pixels : pandas df
        all the pixels of the celestial sphere.
    data : pandas df
        data pixels.
    footprint : str
        footprint name.
    outName : str
        output file name (full path).

    Returns
    -------
    contour : pandas df
        df resu (two cols: footprint and healpixID).

    """
    r = []
    delta_ra = 1.5
    ras = np.arange(0., 360., delta_ra)
    
    for ira in range(len(ras)):
        r += match_pixels(all_pixels, data, ras[ira], delta_ra)

    r = list(map(int, r))
    
    contour = pd.DataFrame(r, columns=['healpixID'])
    #contour = pix_coord(contour)
    contour['footprint'] = footprint
    
    contour.to_hdf(outName,key='contour')
    return contour


inputDir = 'input/cosmology/footprints'
footprints = ['TiDES', 'desi_v3', '4hs_v3', 'desiext_v3', 'crs_v3','desi2_v3']

outDir = 'input/cosmology/contours'


all_pixels = get_all_pixels()

df = pd.DataFrame()
for footprint in footprints:

    data = pd.read_hdf(path(inputDir, footprint))
    data = pix_coord(data)
    outName= '{}/footprint_{}_WFD.hdf5'.format(outDir,footprint)
    dfb = get_contours(all_pixels,data,footprint,outName)



"""
dx = 5
dy = 5
x = np.arange(0., 360., dx)
y = np.arange(-90., 90., dy)


cont = Contours(x, dx, y, dy)
bb = cont(pix_coord(data)[['pixRA', 'pixDec']])
bb.to_hdf('for_contour.hdf5', key='tt')
print(bb)


bb = pd.read_hdf('for_contour.hdf5')
print(bb)

hpxmap = np.full((len(x), len(y)), 0)

print(hpxmap.shape)
hpxmap[np.array(bb['i'].to_list()), np.array(bb['j'].to_list())] = 110000.
contours = measure.find_contours(hpxmap, 0.8)


fig, ax = plt.subplots()
# ax.imshow(hpxmap, cmap=plt.cm.jet)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()
"""