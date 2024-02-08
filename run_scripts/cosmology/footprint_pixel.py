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
from sn_tools.sn_visu import get_map, plot_pixels


def point_in_triangle(point, triangle):
    """Returns True if the point is inside the triangle
    and returns False if it falls outside.
    - The argument *point* is a tuple with two elements
    containing the X,Y coordinates respectively.
    - The argument *triangle* is a tuple with three elements each
    element consisting of a tuple of X,Y coordinates.

    It works like this:
    Walk clockwise or counterclockwise around the triangle
    and project the point onto the segment we are crossing
    by using the dot product.
    Finally, check that the vector created is on the same side
    for each of the triangle's segments.
    """
    # Unpack arguments
    # x, y = point
    x = np.array(point)[:, 0]
    y = np.array(point)[:, 1]

    ax, ay = triangle[0]
    bx, by = triangle[1]
    cx, cy = triangle[2]
    # Segment A to B
    side_1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
    # Segment B to C
    side_2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
    # Segment C to A
    side_3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
    # All the signs must be positive or all negative

    vvneg = (side_1 < 0.0) & (side_2 < 0.0) & (side_3 < 0.0)
    vvpos = (side_1 > 0.0) & (side_2 > 0.0) & (side_3 > 0.0)

    return np.logical_or(vvneg, vvpos)


def tag_Tides_MW(map_pixel) -> pd.DataFrame:
    """
    Function to exclude (weight=-1) MW to get TiDES footprint

    Parameters
    ----------
    map_pixel : pandas df
        original pixel map.

    Returns
    -------
    map_pixel : pandas df
        output pixel map.

    """

    # exclude MW zone
    xc = 0.8*360/24+180.
    yc = 20.
    radius_1 = 60+yc
    radius_2 = 75+yc

    map_pixel['pix_RA_mod'] = map_pixel['pix_RA']
    idx = map_pixel['pix_RA_mod'] >= 180.
    map_pixel.loc[idx, 'pix_RA_mod'] -= 0.
    map_pixel['dist'] = (map_pixel['pix_RA_mod']-xc)**2 + \
        (map_pixel['pix_Dec']-yc)**2

    idxa = map_pixel['dist'] < radius_1**2
    sela = map_pixel[idxa]
    idxb = map_pixel['dist'] < radius_2**2
    selb = map_pixel[idxb]
    idxc = selb['healpixID'].isin(sela['healpixID'].to_list())
    ll = selb.loc[~idxc, 'healpixID'].to_list()

    idx = map_pixel['healpixID'].isin(ll)
    map_pixel.loc[idx, 'weight'] = -1

    map_pixel = map_pixel.drop(columns=['pix_RA_mod'])

    return map_pixel


def tag_Tides_zone_1(map_pixel) -> pd.DataFrame:
    """
    Function to exclude (weight=-1) a zone around -40 deg in Dec
    to get TiDES footprint

    Parameters
    ----------
    map_pixel : pandas df
        original pixel map.

    Returns
    -------
    map_pixel : pandas df
        output pixel map.

    """

    # exclude MW zone
    xc = 0.8*360/24+180.
    yc = 17.14
    radius = 43.43+yc

    map_pixel['pix_RA_mod'] = map_pixel['pix_RA']
    idx = map_pixel['pix_RA_mod'] >= 180.
    map_pixel.loc[idx, 'pix_RA_mod'] -= 0.
    map_pixel['dist'] = (map_pixel['pix_RA_mod']-xc)**2 + \
        (map_pixel['pix_Dec']-yc)**2

    idxa = map_pixel['dist'] < radius**2
    idxa &= map_pixel['pix_Dec'] <= -35.4

    map_pixel.loc[idxa, 'weight'] = -1

    map_pixel = map_pixel.drop(columns=['pix_RA_mod'])

    return map_pixel


def tag_Tides_upper_zone(map_pixel) -> pd.DataFrame:
    """
    Function to exclude (weight=-1) a some areas
    to get TiDES footprint

    Parameters
    ----------
    map_pixel : pandas df
        original pixel map.

    Returns
    -------
    map_pixel : pandas df
        output pixel map.

    """

    #idx = map_pixel['pix_RA'] <= 218
    idx = map_pixel['pix_RA'] >= 110+180
    idx &= map_pixel['pix_Dec'] > 0.
    map_pixel.loc[idx, 'weight'] = -1

    idx = map_pixel['pix_RA'] <= 265.
    idx &= map_pixel['pix_RA'] >= 120.
    idx &= map_pixel['pix_Dec'] > 12.
    map_pixel.loc[idx, 'weight'] = -1

    idx = map_pixel['pix_RA'] <= 180.+27.6
    idx &= map_pixel['pix_RA'] >= 180.-35
    idx &= map_pixel['pix_Dec'] > 6.
    map_pixel.loc[idx, 'weight'] = -1

    """
    idx = map_pixel['pix_RA'] <= 26.+180
    idx &= map_pixel['pix_Dec'] > 6
    map_pixel.loc[idx, 'weight'] = -1

    idx = map_pixel['pix_RA'] >= 360.-26-180.
    idx &= map_pixel['pix_Dec'] > 6
    map_pixel.loc[idx, 'weight'] = -1
    
    """

    triangle = []
    triangle.append((218.-180., 12.))
    triangle.append((218.-180., -12))
    triangle.append((250-180., 12))

    ll = np.c_[map_pixel['pix_RA'].to_numpy(
    ), map_pixel['pix_Dec'].to_list()].tolist()

    x = np.array(ll)[:, 0].tolist()

    tt = point_in_triangle(ll, triangle)

    map_pixel['in_triangle'] = tt
    idx = map_pixel['in_triangle'] == True
    map_pixel.loc[idx, 'weight'] = -1

    idx = map_pixel['pix_RA'] <= 39.
    idx &= map_pixel['pix_Dec'] > 0.
    map_pixel.loc[idx, 'weight'] = -1

    idx = map_pixel['pix_RA'] >= 39.
    idx &= map_pixel['pix_RA'] <= 93.
    idx &= map_pixel['pix_Dec'] > 12.
    map_pixel.loc[idx, 'weight'] = -1

    """
    idx = map_pixel['pix_RA'] >= 218
    idx = map_pixel['pix_RA'] <= 272
    idx &= map_pixel['pix_Dec'] > 12
    map_pixel.loc[idx, 'weight'] = -1
    """

    map_pixel = map_pixel.drop(columns=['in_triangle'])

    return map_pixel


def footprint_TiDES(nside):
    """
    Function to estimte TiDES footprint

    Parameters
    ----------
    nside : int
        healpix nside param.

    Returns
    -------
    None.

    """

    map_pixel = get_map(nside)

    # plot_pixels(map_pixel)

    # flag dec < - 67 deg
    idx = map_pixel['pix_Dec'] < -80.0
    map_pixel.loc[idx, 'weight'] = -1
    # plot_pixels(map_pixel)

    # flag Dec > 17.4

    idx = map_pixel['pix_Dec'] > 17.4
    map_pixel.loc[idx, 'weight'] = -1
    #plot_pixels(map_pixel, rot=(180., 0., 0.))

    # Tides MW
    map_pixel = tag_Tides_MW(map_pixel)

    #plot_pixels(map_pixel, rot=(180., 0., 0.))

    map_pixel = tag_Tides_zone_1(map_pixel)
    #plot_pixels(map_pixel, rot=(180., 0., 0.))

    map_pixel = tag_Tides_upper_zone(map_pixel)

    # select ony pixels of the footprint
    save_footprint(map_pixel, 'TiDES_WFD')

    plot_pixels(map_pixel, rot=(180., 0., 0.))

    plot_pixels(map_pixel)
    plt.show()


def footprint_points(nside, fcoord) -> pd.DataFrame:
    """
    Function to get pixels around a set of (RA,Dec) positions

    Parameters
    ----------
    nside : int
        healpix nside parameter.
    fcoord : dict
        coordinates.

    Returns
    -------
    map_pixel : pandas df
        resulting pixel map.

    """

    map_pixel = get_map(nside)

    hpixes = []
    radius = np.deg2rad(2.52)  # 20 deg2 FP
    for key, vv in fcoord.items():
        tl = hp.ang2pix(nside, vv[0], vv[1], nest=True, lonlat=True)
        hpixes.append(tl)
        vec = hp.pix2vec(nside, tl, nest=True)
        ll = hp.query_disc(nside, vec, radius, nest=True)
        hpixes += ll.tolist()

    idx = map_pixel['healpixID'].isin(hpixes)

    map_pixel.loc[~idx, 'weight'] = -1

    return map_pixel


def footprint_DDF(nside=128):
    """
    Function to estimate DDF footprint

    Parameters
    ----------
    nside : int, optional
        nside healpix param. The default is 128.

    Returns
    -------
    None.

    """

    map_pixel = get_map(nside)

    fields = ['COSMOS', 'XMM-LSS']
    coords = [(150.1, 2.1965), (35.72, -4.75)]

    fcoord = dict(zip(fields, coords))

    map_pixel = footprint_points(nside, fcoord)

    save_footprint(map_pixel, 'Subaru_DDF')
    plot_pixels(map_pixel)

    fields = ['ECFDS', 'ELAISS1']
    coords = [(53.16, -28.095), (9.467, -44.016)]

    fcoord = dict(zip(fields, coords))

    map_pixel = footprint_points(nside, fcoord)
    save_footprint(map_pixel, 'TiDES_DDF')
    plot_pixels(map_pixel)

    plt.show()


def footprint_DESI(fName, outName):
    """
    Function to estimate and save DESI/DESIII foorptints

    Parameters
    ----------
    fName : str
        Footprint file name.
    outName : str
        Tag for outname.

    Returns
    -------
    None.

    """

    hpx_map_desi = np.load(fName)
    hp.mollview((hpx_map_desi), hold=True, nest=True, fig=1)
    hp.graticule()

    nside = 64
    npix = hp.nside2npix(nside)
    index = np.arange(npix)
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    ra, dec = np.degrees(np.pi*2.-phi), -np.degrees(theta-np.pi/2.)

    vec = hp.pix2ang(nside, range(npix), nest=True, lonlat=True)

    map_pixel = get_map(nside)

    map_pixel['weight'] = hpx_map_desi
    idx = map_pixel['weight'] == 0
    map_pixel.loc[idx, 'weight'] = -1

    plot_pixels(map_pixel)

    save_footprint(map_pixel, '{}_WFD'.format(outName))
    plt.show()


def save_footprint(map_pixel, footprint):
    """
    Function to dump pixela with positive weight on disk

    Parameters
    ----------
    map_pixel : pandas df
        Data to dump.
    footprint : str
        footprint name.

    Returns
    -------
    None.

    """

    idx = map_pixel['weight'] > 0
    sel = pd.DataFrame(map_pixel[idx])

    sel['footprint'] = footprint

    tt = sel[['footprint', 'healpixID']]

    tt.to_hdf('footprint_{}.hdf5'.format(footprint), key='footprint')


parser = OptionParser()
parser.add_option("--nside", type=int, default=64,
                  help="nside healpix parameter[%default]")
parser.add_option("--footprint_type", type=str, default='WFD',
                  help="footprint type (WFD/DDF)[%default]")

opts, args = parser.parse_args()

nside = opts.nside
foot_type = opts.footprint_type

if foot_type == 'WFD':
    footprint_TiDES(nside)

if foot_type == 'DDF':
    footprint_DDF(nside)

if foot_type == 'DESI':
    footprint_DESI('desi_footprint_bright.npy', foot_type)

if foot_type == 'DESI2':
    footprint_DESI('desi2_footprint_bright.npy', foot_type)
