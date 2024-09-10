import time
from sn_tools.sn_obs import proj_gnomonic_plane
from sn_tools.sn_obs import LSSTPointing_circular
from sn_tools.sn_utils import multiproc
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from descartes.patch import PolygonPatch
from shapely import affinity
import pandas as pd
import healpy as hp


def get_pixels(RA, Dec, nside=64, widthRA=5.):
    """
    grab pixels around (RA,Dec) (window width: widthRA)

    Parameters
    ----------
    RA : float
        RA value.
    Dec : float
        Dec value.
    nside : int, optional
        nside healpix parameter. The default is 64.
    widthRA : float, optional
        window width. The default is 5..

    Returns
    -------
    healpixIDs : int
        healpixIDs.
    float
        pixRA.
    float
        pixDec.

    """

    healpixID = hp.ang2pix(nside, RA, Dec, nest=True, lonlat=True)
    vec = hp.pix2vec(nside, healpixID, nest=True)
    healpixIDs = hp.query_disc(nside, vec, np.deg2rad(widthRA),
                               inclusive=True, nest=True)

    # get pixel coordinates
    coords = hp.pix2ang(nside, healpixIDs, nest=True, lonlat=True)

    return healpixIDs, coords[0], coords[1]


def get_xy(RA, Dec, nside=64):
    """
    Grab gnomonic projection of pixels around(RA,Dec)

    Parameters
    ----------
    RA : float
        RA value.
    Dec : float
        Dec value.
    nside: int, opt.
        nside healpix value. The default is 64.

    Returns
    -------
    x : float
        x-axis values.
    y : float
        y-axis values.

    """
    healpixID, pixRA, pixDec = get_pixels(RA, Dec, nside=nside)

    # print(pixRA, pixDec)
    pixRA_rad = np.deg2rad(pixRA)
    pixDec_rad = np.deg2rad(pixDec)
    # convert data position in rad
    # pRA = np.median(sel_data['RA'])
    # pDec = np.median(sel_data['Dec'])
    pRA_rad = np.deg2rad(RA)
    pDec_rad = np.deg2rad(Dec)

    # gnomonic projection of pixels on the focal plane
    x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, pixRA_rad, pixDec_rad)

    df = pd.DataFrame(healpixID, columns=['healpixID'])
    df['pixRA'] = pixRA
    df['pixDec'] = pixDec
    df['xpixel_norot'] = x
    df['ypixel_norot'] = y

    return df


class FocalPlane:
    def __init__(self, nx=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 8*15])),
                 ny=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 2*15])),
                 FoV=9.62,
                 level='raft',
                 ccd_sub=dict(zip(['to_remove', 'guide', 'sensor'],
                                  [['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                                    '1_13', '1_14', '1_15', '2_14', '2_15', '3_15',
                                    '15_1', '15_2', '15_3', '14_1', '14_2', '13_1',
                                    '15_13', '15_14', '15_15', '14_14', '14_15', '13_15'],
                              ['2_3', '3_2', '2_13', '3_14',
                                      '14_3', '13_2', '14_13', '13_14'],
                              ['3_3', '3_13', '13_3', '13_13']]))):
        """
        Focal Plane class

        Parameters
        ----------
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
        ccd_sub :dict, optional
            list of ccds to remove.
            The default is dict(zip(['to_remove', 'guide', 'sensor'],
                            [['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                            '1_13', '1_14', '1_15', '2_14', '2_15', '3_15',
                            '15_1', '15_2', '15_3', '14_1', '14_2', '13_1',
                            '15_13', '15_14', '15_15', '14_14', '14_15', '13_15'],
                            ['2_3', '3_2', '2_13', '3_14',
                             '14_3', '13_2', '14_13', '13_14'],
                            ['3_3', '3_13', '13_3', '13_13']])).

        Returns
        -------
        None.

        """

        fov_str = FoV*(np.pi/180.)**2  # LSST fov in sr
        theta = 2.*np.arcsin(np.sqrt(fov_str/(4.*np.pi)))
        fpscale = np.tan(theta)

        self.xmin, self.xmax = -fpscale, fpscale
        self.ymin, self.ymax = -fpscale, fpscale

        self.dx = self.xmax-self.xmin
        self.dy = self.ymax-self.ymin

        self.level = level
        self.nx = nx
        self.ny = ny
        self.index_level = ['raft', 'ccd', 'sensor']
        self.ccd_sub = ccd_sub

        self.fp = self.buildIt()

        for key, vals in ccd_sub.items():
            self.remove_ccd(vals)

        self.ccols = ['healpixID', 'pixRA', 'pixDec',
                      'observationId', 'raft']
        if level == 'ccd':
            self.ccols.append('ccd')
        if level == 'sensor':
            self.ccols.append('ccd')
            self.ccols.append('sensor')

    def set_display_mode(self):

        self.ccols += ['xpixel', 'ypixel', 'xmin', 'xmax', 'ymin', 'ymax']

    def buildIt(self):
        """
        Build the FP here

        Returns
        -------
        pandas df
            resulting FP.

        """

        d_elem_x = self.dx/self.nx[self.level]
        d_elem_y = self.dy/self.ny[self.level]

        xvalues = np.arange(self.xmin, self.xmax, d_elem_x)
        yvalues = np.arange(self.ymin, self.ymax, d_elem_y)
        xv, yv = np.meshgrid(xvalues, yvalues)

        df_fp = pd.DataFrame(xv.flatten(), columns=['x'])
        df_fp['y'] = yv.flatten()
        df_fp['xc'] = df_fp['x']+0.5*d_elem_x
        df_fp['yc'] = df_fp['y']+0.5*d_elem_y
        df_fp['xmin'] = df_fp['xc']-0.5*d_elem_x
        df_fp['xmax'] = df_fp['xc']+0.5*d_elem_x
        df_fp['ymin'] = df_fp['yc']-0.5*d_elem_y
        df_fp['ymax'] = df_fp['yc']+0.5*d_elem_y

        # get index here

        idx_level = self.index_level.index(self.level)
        for i in range(idx_level+1):
            vv = self.index_level[i]
            d_elem_xx = self.dx/self.nx[vv]
            d_elem_yy = self.dy/self.ny[vv]
            df_fp = self.get_index(df_fp, d_elem_xx, d_elem_yy, vv)

        # remove extra cells
        idx = df_fp['xc'] > self.xmin
        idx &= df_fp['xc'] < self.xmax
        idx &= df_fp['yc'] > self.ymin
        idx &= df_fp['yc'] < self.ymax

        return pd.DataFrame(df_fp[idx])

    def get_index(self, df_fp, d_elem_x, d_elem_y, level):
        """
        Method to estimate cells index from position

        Parameters
        ----------
        df_fp : pandas df
            Data to process.
        d_elem_x : float
            x-axis cell size.
        d_elem_y : float
            y-axis cell size.
        level : str
            segmentation level.

        Returns
        -------
        df_fp : pandas df
            original df+index.

        """

        ipos = (df_fp['xc']-self.xmin)/d_elem_x+1
        jpos = (self.ymax-df_fp['yc'])/d_elem_y+1
        lpj = '{}_j'.format(level)
        lpi = '{}_i'.format(level)
        df_fp[lpj] = ipos
        df_fp[lpi] = jpos
        df_fp[lpj] = df_fp[lpj].astype(int)
        df_fp[lpi] = df_fp[lpi].astype(int)
        df_fp[level] = df_fp[lpi].astype(str)+'_'+df_fp[lpj].astype(str)
        df_fp = df_fp.drop(columns=[lpi, lpj])

        return df_fp

    def check_fp(self, top_level='raft', low_level='ccd'):
        """
        Method to estimate some infos on the focal plane

        Parameters
        ----------
        top_level : str, optional
            top-level to get infos. The default is 'raft'.
        low_level : str, optional
            lower level for infos. The default is 'ccd'.

        Returns
        -------
        None.

        """

        posi = '{}_i'.format(top_level)
        posj = '{}_j'.format(top_level)

        ijpos = self.fp[top_level].unique()

        print('Number of {}s'.format(top_level), len(ijpos))
        if low_level != top_level:
            for vv in ijpos:
                idx = self.fp[top_level] == vv
                sel = self.fp[idx]
                idxb = sel[low_level].unique()
                print('{}'.format(top_level), vv,
                      'N {}s'.format(low_level), len(idxb))

    def remove_ccd(self, ccdlist):
        """
        Method to remove ccds from fp

        Parameters
        ----------
        ccdlist : list(str)
            List of ccds to remove.

        Returns
        -------
        None.

        """

        idx = self.fp['ccd'].isin(ccdlist)

        self.fp = self.fp[~idx]

    def plot_fp(self):
        """
        Method to plot the focal plane

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots()
        ax.plot(self.fp['x'], self.fp['y'], 'k.')
        ax.plot(self.fp['xc'], self.fp['yc'], 'r*')

        ax.plot(self.fp['xmin'], self.fp['ymin'], color='b',
                marker='s', mfc='None', linestyle='None')
        ax.plot(self.fp['xmin'], self.fp['ymax'], color='b',
                marker='s', mfc='None', linestyle='None')
        ax.plot(self.fp['xmax'], self.fp['ymin'], color='b',
                marker='s', mfc='None', linestyle='None')
        ax.plot(self.fp['xmax'], self.fp['ymax'], color='b',
                marker='s', mfc='None', linestyle='None')

        plt.show()

    def plot_fp_pixels(self, pixels=None, signal=None):
        """
        Method to plot the FP pixels and pixels inside

        Parameters
        ----------
        pixels : pandas df, optional
            Pixel coordinates. The default is None.
        signal : pandas df, optional
            FP pixels with pixels (healpiX) inside. The default is None.

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(10, 10))

        # draw the focal plane

        for i, row in self.fp.iterrows():

            rect = self.get_rect(row)
            ax.add_patch(rect)

        xmin = self.fp['xmin'].min()
        xmax = self.fp['xmax'].max()
        ymin = self.fp['ymin'].min()
        ymax = self.fp['ymax'].max()

        k = 1.5
        ax.set_xlim([k*xmin, k*xmax])
        ax.set_ylim([k*ymin, k*ymax])

        if pixels is not None:
            ax.plot(pixels['xpixel'], pixels['ypixel'], 'r.')

        if signal is not None:
            for i, row in signal.iterrows():
                rect = self.get_rect(row, fill=True)
                ax.add_patch(rect)

        plt.show()

    def get_rect(self, row, fill=False):

        from matplotlib.patches import Rectangle
        xy = (row['xmin'], row['ymin'])
        height = row['ymax']-row['ymin']
        width = row['xmax']-row['xmin']
        rect = Rectangle(xy, width, height, fill=fill)

        return rect

    def pix_to_obs(self, df_pix):

        # make super df

        df_super = self.fp.merge(df_pix, how='cross')
        print(df_super)
        # select pixels inside FP
        idx = df_super['xpixel'] >= df_super['xmin']
        idx &= df_super['xpixel'] <= df_super['xmax']
        idx &= df_super['ypixel'] >= df_super['ymin']
        idx &= df_super['ypixel'] <= df_super['ymax']

        res = pd.DataFrame(df_super[idx])

        # delete df_super
        return res[self.ccols]


def get_simuData(dbDir, dbName):
    """
    To load simulation data

    Parameters
    ----------
    dbDir : str
        Db loc dir.
    dbName : str
        Db name.

    Returns
    -------
    data : numpy array
        Data loaded.

    """

    fName = '{}/{}'.format(dbDir, dbName)

    data = np.load(fName, allow_pickle=True)

    return data


def get_proj_data(sel_data, nside=64):
    """
    Function to get gnomonic projection of a pixel corresponding
    to a set of pointings.

    Parameters
    ----------
    sel_data : pandas df
        Data to process.
    nside: int, opt.
        healpix nside parameter. The default is 64.

    Returns
    -------
    df_pix : pandas df
        projected pixels.

    """

    df_pix = pd.DataFrame()
    for vv in sel_data:
        dd = get_xy(vv['fieldRA'], vv['fieldDec'], nside=nside)
        """
        dd = pd.DataFrame(x, columns=['xpixel_norot'])
        dd['ypixel_norot'] = y
        """
        for var in ['observationId', 'filter', 'rotSkyPos']:
            dd[var] = vv[var]
        df_pix = pd.concat((df_pix, dd))

    # pixel rotation here
    df_pix['rotSkyPixel'] = -np.deg2rad(df_pix['rotSkyPos'])
    # df_pix['rotSkyPixel'] = 0.
    df_pix['xpixel'] = np.cos(df_pix['rotSkyPixel'])*df_pix['xpixel_norot']
    df_pix['xpixel'] -= np.sin(df_pix['rotSkyPixel'])*df_pix['ypixel_norot']
    df_pix['ypixel'] = np.sin(df_pix['rotSkyPixel'])*df_pix['xpixel_norot']
    df_pix['ypixel'] += np.cos(df_pix['rotSkyPixel'])*df_pix['ypixel_norot']

    return df_pix


def process_pointings(data, params, j=0, output_q=None):

    nside = params['nside']
    df_fp = params['df_fp']

    time_ref = time.time()
    print('start processing', j, len(data))
    df_pix = get_proj_data(data, nside=nside)
    print('booooo', df_pix)
    # get matching pixels<-> FP pos
    # df_fp.set_display_mode() #mandatory if plot_fp_pixels is to be used
    pix_obs = df_fp.pix_to_obs(df_pix)

    print('processing', j, time.time()-time_ref)
    if output_q is not None:
        return output_q.put({j: pix_obs})
    else:
        return pix_obs


FoV = 9.62  # area in deg2

fov_str = FoV*(np.pi/180.)**2  # LSST fov in sr
theta = 2.*np.arcsin(np.sqrt(fov_str/(4.*np.pi)))
fpscale = np.tan(theta)

print('alll', fpscale)

pointing = LSSTPointing_circular(0., 0., maxbound=fpscale)

bounds = pointing.boundary.coords[:]

print(type(bounds))

tt = np.rec.fromrecords(bounds, names=['x', 'y'])

print(tt['x'])

plt.plot(tt['x'], tt['y'], 'k.')

xmin = np.min(tt['x'])
xmax = np.max(tt['x'])

print('allo', fpscale/xmax, xmin, xmax)
# print(test)
d = xmax-xmin
nx = 15
ny = int(nx/2)
ymin = 0.
d_elem = float(d/nx)

rangey = range(-ny, ny+1)

num = 0
pixels = {}
xmax += d_elem

ny = nx
ymin = xmin
ymax = xmax
# xvalues = np.linspace(xmin, xmax, nx, endpoint=False)
# yvalues = np.linspace(ymin, ymax, ny, endpoint=False)

df_fp = FocalPlane(level='ccd')

df_fp.check_fp(top_level='raft', low_level='ccd')


simu_data = get_simuData('../DB_Files', 'baseline_v3.0_10yrs.npy')

print(len(simu_data))
idx = simu_data['note'] == 'DD:COSMOS'
sel_data = simu_data[idx]
print('data to process', len(sel_data))

nside = 64
params = {}
params['nside'] = 128
params['df_fp'] = df_fp
# get proj pixels for obs
time_ref = time.time()
pix_obs = multiproc(sel_data, params, process_pointings, 8)

# print(len(df_fp.fp), len(df_pix), len(df_super), len(df_super[idx]))
print('elapse time', time.time()-time_ref)

# df_fp.plot_fp_pixels(pixels=df_pix, signal=pix_obs)


print(test)
for i in range(nx):
    xa = xmin+i*d_elem
    xb = xmin+(i+1)*d_elem

    ya = ymin-0.5*d_elem
    yb = ymin+0.5*d_elem
    num += 1
    r = []
    r.append((xa, ya))
    r.append((xa, yb))
    r.append((xb, yb))
    r.append((xb, ya))

    polya = geometry.Polygon(r)
    pixels[num] = polya

    # translate
    for k in rangey:
        num += 1
        polyc = affinity.translate(
            polya, xoff=0.0, yoff=polya.centroid.y-k*d_elem)
        pixels[num] = polyc

ttb = np.rec.fromrecords(r, names=['x', 'y'])

fig, ax = plt.subplots()
ax.plot(tt['x'], tt['y'], 'k.')
# ax.plot(ttb['x'], ttb['y'], 'r*', linestyle='None')

for key, val in pixels.items():
    pf = PolygonPatch(val, facecolor=(0, 0, 0, 0), edgecolor='red')
    ax.add_patch(pf)


plt.show()
