#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:17:54 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import numpy as np
from sn_tools.sn_fp_pixel import get_pixels_in_window,FocalPlane,get_xy_pixels
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
from astropy.time import Time
from sn_scheduler.scheduler import StarAltTime
import astropy.units as u
import pandas as pd

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
        
        
def plotMollview(pixels, axa, fig, night,nside,comment=''):
    """
    Method to display a Mollweid view

    Parameters
    --------------
    pixels: pandas df
      data to plots
    axa: matplotlib axis
      axis to use for the plot
    fig: matplotlib figure
      figure to use for plot
    night: int
      night number
    """

    xmin = 0.99999
    xmax = np.max([np.max(pixels['color']), 1])

    norm = plt.cm.colors.Normalize(xmin, xmax)
    # cmap = plt.get_cmap('jet', int(xmax))
    n = int(xmax)+1
    n = 6
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cmap = from_list(None, plt.cm.Set1(range(1, n)), n-1)
    cmap.set_under('w')

    npixels = hp.nside2npix(nside)
    hpxmap = np.zeros(npixels, dtype=int)
    hpxmap = np.full(hpxmap.shape, -2)
    hpxmap[pixels['healpixID']] = pixels['color'].astype(int)

    # print('hello ',xmin,xmax)

    dd = '$\Delta$T = current night-last obs night (gri) [days]'
    hp.mollview(hpxmap, nest=True, cmap=cmap,
                min=xmin, max=n, norm=norm, cbar=False,
                title=comment, hold=True, badcolor='white', xsize=1600)

    hp.graticule(verbose=False)
    
    """
    ax = plt.gca()
    image = ax.get_images()[0]
    cbar = fig.colorbar(image, ax=ax, ticks=range(
        1, n), orientation='horizontal')
    # cbar = fig.colorbar(ax[0,0], ticks=range(0,n), orientation='horizontal')  # set some values to ticks

    labels = list(range(1, n))

    tick_label = list(map(str, labels))

    tick_label[-1] = '>{}'.format(tick_label[-2])
    # print(tick_label)
    cbar.ax.set_xticklabels([])
    cbar.ax.tick_params(size=0)
    for j, lab in enumerate(tick_label):
        cbar.ax.text(labels[j]+0.5, -10., lab)

    #ax.text(-3.5, 0.9, self.dbName, fontsize=15, color='r')
    ax.text(-3.5, 0.6, 'night {}'.format(night), fontsize=15, color='k')        
    """

def process_night(stars_alt, year, month, day, targets, plot_it=False):
    """


    Parameters
    ----------
    stars_alt : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    plot_it : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    targets_info : TYPE
        DESCRIPTION.

    """
    """
    Function to process targets

    Parameters
    ----------
    stars_alt : StarAltTime instance
        The class where the calculation is made.
    year : int
        year of observation.
    month : int
        month of observation.
    day : int
        day of observation.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    targets_info : pandas df
        Targets with obs. info.

    """

    # grab the targets
    stars_alt.target_location(targets=targets)

    # get stars alt
    stars_alt(year=year, month=month, day=day)

    # grab star infos
    alt_min = 25.
    alt_max = 86.5
    airmass_max = 2.5

    targets_info = stars_alt.target_info(star_alt_min=alt_min*u.deg,
                                         star_alt_max=alt_max*u.deg,
                                         star_airmass_max=airmass_max)

    """
    print(targets_info.columns)
    print(targets_info[['target', 'mjd', 'mjd_per_min_p1',
          'mjd_per_max_p1', 'obs_duration [h]']])
    """
    # plot result here
    if plot_it:
        stars_alt.plot(star_alt_min=alt_min*u.deg,
                       star_alt_max=alt_max*u.deg,
                       star_airmass_max=airmass_max)
        # stars_alt.plot_airmass()
        plt.show()

    return targets_info

def make_df(target,ra,dec):
    
    df = pd.DataFrame(target,columns=['target'])
    df['ra'] = ra
    df['dec'] = dec
    
    return df
    
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

#band index
bbands = dict(zip('ugrizy',[1,2,3,4,5,6]))

## StarAltTime instance
stars_alt = StarAltTime()
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
        band = dd['filter']
        mjd = np.round(dd['mjd'],3)
        field = dd['target_name'].split(':')[-1]
        ttime = Time('{}'.format(mjd),format='mjd')
        tdate = '{}'.format(ttime.datetime64)
        spa = tdate.split('T')[0].split('-')
        year=int(spa[0])
        month = int(spa[1])
        day = int(spa[2])
        print(year,month,day)
        
        ppixels = pix_in_fp(dd,RA,Dec)
    
        #pix_in_fp.plot_pixels_in_FP(ppixels)
        
        ppixels = ppixels[df_var]
        
        print(ppixels,len(ppixels),band)
        
        fig, ax = plt.subplots(figsize=(12,8))
        
        ppixels['color'] = bbands[band]
        comment = '{},night={},mjd={},{}-band'.format(field,night,mjd,band)
        plotMollview(ppixels, ax, fig, night, nside,comment)
        
        targets = make_df([field],[RA], [Dec])
                            
        rr = process_night(stars_alt, year, month, day, targets, plot_it=True)
        
        plt.show()
        
        
    print(test)