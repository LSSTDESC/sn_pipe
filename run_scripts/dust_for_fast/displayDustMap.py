from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 12


def getDustMap(nside=64):
    """
    Method to estimate the dust map 

    Parameters
    ---------------
    nside: int, opt
      healpix nside parameters

    Returns
    -----------
    dustmap: pandas df
    with the following columns: healpixID, RA, Dec, ebvofMW

    """

    # get the total number of pixels
    npixels = hp.nside2npix(nside)

    # pixels = hp.get_all_neighbours(nside, 0.0, 0.0, nest=True, lonlat=Tru)

    # get the (RA, Dec) of the pixel centers
    vec = hp.pix2ang(nside, range(npixels), nest=True, lonlat=True)

    dustmap = pd.DataFrame(range(npixels), columns=['healpixID'])
    dustmap['RA'] = vec[0]
    dustmap['Dec'] = vec[1]

    coords = SkyCoord(vec[0], vec[1], unit='deg')
    try:
        sfd = SFDQuery()
    except Exception as err:
        from dustmaps.config import config
        config['data_dir'] = 'dustmaps'
        import dustmaps.sfd
        dustmaps.sfd.fetch()
    sfd = SFDQuery()
    ebvofMW = sfd(coords)

    dustmap['ebvofMW'] = ebvofMW
    dustmap['npixels'] = npixels

    return dustmap


def plotDustMap(dustmap, ebvofMW_cut=0.25):
    """
    Function to plot the dustmap (ebvofMW) in Mollweid view

    Parameters
    ---------------
    dustmap: pandas df
      data to display
    ebvofMW_cut: float, opt
       cut for the display (default: 0.25)

    """
    npixels = np.unique(dustmap['npixels'])
    idx = dustmap['ebvofMW'] <= ebvofMW_cut

    dustmap = dustmap[idx]

    xmin = 0.00001
    xmax = ebvofMW_cut

    norm = plt.cm.colors.Normalize(xmin, xmax)
    cmap = plt.cm.jet
    cmap.set_under('w')

    hpxmap = np.zeros(npixels, dtype=np.float)
    hpxmap = np.full(hpxmap.shape, 0.)
    hpxmap[dustmap['healpixID']] += dustmap['ebvofMW']

    hp.mollview(hpxmap, cmap=cmap, nest=True,
                min=xmin, max=xmax, norm=norm,
                title='E(B-V) MW - SFD')
    hp.graticule()


def plotDustHist(dustmap, ebvofMW_cut=0.25):
    """
    Function to plot the dustmap (ebvofMW) in Mollweid view

    Parameters
    ---------------
    dustmap: pandas df
      data to display
    ebvofMW_cut: float, opt
       cut for the display (default: 0.25)

    """
    npixels = np.unique(dustmap['npixels'])
    idx = dustmap['ebvofMW'] <= ebvofMW_cut

    dustmap = dustmap[idx]

    fig, ax = plt.subplots()
    ax.hist(dustmap['ebvofMW'], histtype='step', bins=100)
    ax.set_xlabel('E(B-V)')
    ax.set_ylabel('Number of Entries')


nside = 128

dustmap = getDustMap(nside=nside)

plotDustMap(dustmap)
plotDustHist(dustmap)

#plt.plot(vec[0], vec[1], 'ko')
plt.show()
