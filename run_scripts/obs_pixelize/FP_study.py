from sn_tools.sn_obs import proj_gnomonic_plane
from sn_tools.sn_obs import LSSTPointing_circular
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from descartes.patch import PolygonPatch
from shapely import affinity
import pandas as pd
import healpy as hp


def get_pixels(RA, Dec, nside=64, widthRA=5.):

    healpixID = hp.ang2pix(nside, RA, Dec, nest=True, lonlat=True)
    print('allo', healpixID)
    vec = hp.pix2vec(nside, healpixID, nest=True)
    print('allo', vec)
    healpixIDs = hp.query_disc(nside, vec, np.deg2rad(widthRA),
                               inclusive=True, nest=True)

    # get pixel coordinates
    coords = hp.pix2ang(nside, healpixIDs, nest=True, lonlat=True)

    return healpixIDs, coords[0], coords[1]


def get_xy(RA, Dec):
    healpixID, pixRA, pixDec = get_pixels(RA, Dec)

    print(pixRA, pixDec)
    pixRA_rad = np.deg2rad(pixRA)
    pixDec_rad = np.deg2rad(pixDec)
    # convert data position in rad
    pRA = np.median(sel_data['RA'])
    pDec = np.median(sel_data['Dec'])
    pRA_rad = np.deg2rad(pRA)
    pDec_rad = np.deg2rad(pDec)

    # gnomonic projection of pixels on the focal plane
    x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, pixRA_rad, pixDec_rad)

    return x, y


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


def get_simuData(dbDir, dbName):

    fName = '{}/{}'.format(dbDir, dbName)

    data = np.load(fName)

    return data


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

df_fp = FocalPlane(level='sensor')

df_fp.check_fp(top_level='ccd', low_level='sensor')

# df_fp_new.plot_fp()


simu_data = get_simuData('../DB_Files', 'baseline_v3.4_10yrs.npy')

print(len(simu_data))
idx = simu_data['note'] == 'DD:COSMOS'
sel_data = simu_data[idx][:2]
print(sel_data.dtype)
df_pix = pd.DataFrame()
for vv in sel_data:
    x, y = get_xy(vv['RA'], vv['Dec'])
    dd = pd.DataFrame(x, columns=['xpixel'])
    dd['ypixel'] = y
    dd['observationId'] = vv['observationId']
    df_pix = pd.concat((df_pix, dd))

# make super df

df_super = df_fp.fp.merge(df_pix, how='cross')
print(df_super)
idx = df_super['xpixel'] >= xmin
idx &= df_super['xpixel'] <= xmax
idx &= df_super['ypixel'] >= ymin
idx &= df_super['ypixel'] <= ymax
print(len(df_fp.fp), len(df_pix), len(df_super), len(df_super[idx]))
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
