#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:17:54 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import numpy as np
from sn_tools.sn_fp_pixel import get_pixels_in_window,FocalPlane,get_xy_pixels


class Pixels_in_FP(FocalPlane):
    def __init__(self,nside,deltaRA,deltaDec,
                 nx=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 8*15])),
                 ny=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 2*15])),
                 FoV=9.6,
                 level='raft',
                 raft_sub=dict(
                     zip(['to_remove'], [['1_1', '1_5', '5_1', '5_5']])),
                 ccd_sub=dict(zip(['to_remove', 'guide', 'sensor'],
                                  [['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                                    '1_13', '1_14', '1_15',
                                    '2_14', '2_15', '3_15',
                                    '15_1', '15_2', '15_3', '14_1',
                                    '14_2', '13_1',
                                    '15_13', '15_14', '15_15', '14_14',
                                    '14_15', '13_15'],
                              ['2_3', '3_2', '2_13', '3_14',
                                      '14_3', '13_2', '14_13', '13_14'],
                              ['3_3', '3_13', '13_3', '13_13']]))):
        """
        class to estimate the list of pixels inside the Focal Plane

        Parameters
        ----------
        nside : int
            nside healpix parameter.
        deltaRA : float
            RA width around the pointing center.
        deltaDec : float
            Dec width around the pointing center.
         nx : dict, optional
            x-axis segmentation (level dep.).
            The default is dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 8*15])).
        ny : dict, optional
            y-axis segmentation (level dependent).
            The default is dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 2*15])).
        FoV : float, optional
            Field of view. The default is 9.62.
        level : str, optional
            segmentation level (raft,ccd,sensor). The default is 'raft'.
        raft_sub: dict, optional
            list of rafts to remove
            the default is dict(zip(['to_remove'],['1_1','1_4','5,1','5_5']))
        ccd_sub :dict, optional
            list of ccds to remove.
            The default is dict(zip(['to_remove', 'guide', 'sensor'],
                            [['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                            '1_13', '1_14', '1_15', '2_14', '2_15', '3_15',
                            '15_1', '15_2', '15_3', '14_1', '14_2', '13_1',
                            '15_13', '15_14', '15_15', '14_14', '14_15', 
                            '13_15'],
                            ['2_3', '3_2', '2_13', '3_14',
                             '14_3', '13_2', '14_13', '13_14'],
                            ['3_3', '3_13', '13_3', '13_13']])).
        Returns
        -------
        None.

        """
        super().__init__(nx,ny,FoV,level,raft_sub,ccd_sub)
    
        self.nside = nside
        self.deltaRA = deltaRA
        self.deltaDec = deltaDec
        

    def __call__(self,obs,RA,Dec):
        """
        Main method

        Parameters
        ----------
        obs : array
            array of observations.
        RA : float
            RA pointing.
        Dec : float
            Dec pointing.

        Returns
        -------
        pandas df
            List of pixels inside the FP.

        """
        
        # get pixels around a (RA,Dec) window
        pixel_obs = get_pixels_in_window(self.nside, 
                                          RA-self.deltaRA, RA+self.deltaRA, 
                                          Dec-self.deltaRA, Dec+self.deltaRA)
        
        #gnomonic projection
        pixel_proj = get_xy_pixels(obs,
                           pixel_obs['healpixID'],
                           pixel_obs['pixRA'],
                           pixel_obs['pixDec'],
                           RACol='RA',
                           DecCol='Dec',
                           telrot=True)
        
        #grab pixels that are inside the FP
        pixel_fp = self.pix_to_obs(pixel_proj)
        
        list_pixels_fp = pixel_fp['healpixID'].tolist()
        ido = np.in1d(pixel_proj['healpixID'],list_pixels_fp)
        
        vvar = ['healpixID','xpixel','ypixel']
        pixel_proj=pixel_proj[ido]
        pixel_proj = pixel_proj[vvar]
        
        pixel_fp = pixel_fp.merge(pixel_proj,
                                  left_on=['healpixID'],
                                  right_on=['healpixID'],
                                  suffixes=['',''])
        return pixel_fp
    
    def plot_pixels_in_FP(self, pixelList):

        #plot
        self.plot_fp_pixels(pixelList)
        
        
        



parser = OptionParser(
    description='Script build an AuxTel survey')

parser.add_option('--dbDir', type=str,
                  default='../DB_Files',
                  help='Data dir [%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--nside', type=int,
                  default=128,
                  help='Healpix nside parameter [%default]')
parser.add_option('--deltaRA', type=float,
                  default=10,
                  help='RA width around pointing center [%default]')
parser.add_option('--deltaDec', type=float,
                  default=10,
                  help='Dec width around pointing center [%default]')
parser.add_option('--fp_level', type=str,
                  default='raft',
                  help='FP granularity level (ccd,raft,sensor) [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
nside = opts.nside
deltaRA=opts.deltaRA
deltaDec=opts.deltaDec
fp_level=opts.fp_level

#class Pixels_in_FP
pix_in_fp = Pixels_in_FP(nside, deltaRA, deltaDec,level=fp_level)

df_var = ['healpixID','pixRA','pixDec','raft']

if fp_level=='ccd':
    df_var += ['ccd']

if fp_level == 'sensor':
    df_var += ['ccd','sensor']

#load the data

fName = '{}/{}.npy'.format(dbDir,dbName)

obs = np.load(fName)

nnights = len(np.unique(obs['night']))

print(nnights)

for night in range(1,nnights+1):
    idx = obs['night'] == night
    sel_night = obs[idx]
    
    tt = np.unique(sel_night['target_name']).tolist()
    tt = list(filter(None, tt))

    list_dd = list(filter(lambda x: 'DD' in x, tt))

    # night with no ddf
    if len(list_dd) == 0:
        continue
    
    print(tt)
    print(list_dd)
    
    #select only observations with these DDFs
    idxa = np.in1d(sel_night['target_name'],list_dd)
    
    sel_dd = sel_night[idxa]
    
    print('alors',np.unique(sel_dd['target_name']))
    
    # sort by mjd
    sel_dd = np.sort(sel_dd,order=['mjd'])
    for dd in sel_dd:
        print(dd[['RA','Dec','mjd']])
        RA=dd['RA']
        Dec = dd['Dec']
        
        ppixels = pix_in_fp(dd,RA,Dec)
    
        pix_in_fp.plot_pixels_in_FP(ppixels)
        
        ppixels = ppixels[df_var]
        
        print(ppixels,len(ppixels))
        
        """
        #get pixels around the pointing
        pixel_obs = get_pixels_in_window(nside, 
                                          RA-deltaRA, RA+deltaRA, 
                                          Dec-deltaRA, Dec+deltaRA)
        print('hello',pixel_obs)
        #gnomonic projection
        pixel_proj = get_xy_pixels(dd,
                           pixel_obs['healpixID'],
                           pixel_obs['pixRA'],
                           pixel_obs['pixDec'],
                           RACol='RA',
                           DecCol='Dec',
                           telrot=True)
        
        print(pixel_proj)
        #grab pixels that are inside the FP
        pixel_fp = fp.pix_to_obs(pixel_proj)
        
        list_pixels_fp = pixel_fp['healpixID'].tolist()
        ido = np.in1d(pixel_proj['healpixID'],list_pixels_fp)
        print(pixel_fp)
        #plot
        fp.plot_fp_pixels(pixel_proj[ido])
        """
    print(test)
        
        
    
    """
    print(tt)
    idb = np.core.defchararray.find(tt,'SSO')
    print(idb)
    print(test)
    """