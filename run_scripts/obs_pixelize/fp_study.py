import time
from sn_tools.sn_utils import multiproc
from sn_tools.sn_fp_pixel import FocalPlane
from sn_tools.sn_fp_pixel import get_proj_data, get_xy_pixels
from sn_tools.sn_fp_pixel import get_window, get_pixels_in_window
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from optparse import OptionParser


def get_simuData(dbDir, dbName, fieldName='DD:COSMOS'):
    """
    To load simulation data

    Parameters
    ----------
    dbDir : str
        Db loc dir.
    dbName : str
        Db name.
    fieldName: str, opt
        field to select. The default is DD:COSMOS

    Returns
    -------
    data : numpy array
        Data loaded.

    """

    fName = '{}/{}'.format(dbDir, dbName)

    data = np.load(fName, allow_pickle=True)
    idx = data['scheduler_note'] == fieldName
    sel_data = data[idx]

    return sel_data


def process_pointings(data, params, j=0, output_q=None):
    """
    Function to process pointings

    Parameters
    ----------
    data : pandas df
        Data to process.
    params : dict
        Parameters.
    j : int, optional
        flag for multiproc. The default is 0.
    output_q : multiproc queue, optional
        queue for multiprocessing. The default is None.

    Returns
    -------
    pandas df
        pixels <-> observations

    """

    nside = params['nside']
    fp_level = params['fp_level']
    RACol = params['RACol']
    DecCol = params['DecCol']
    filterCol = params['filterCol']
    df_fp = FocalPlane(level=fp_level)

    # df_fp.check_fp(top_level='raft', low_level='ccd')

    time_ref = time.time()
    print('start processing', j, len(data))
    df_pix = get_proj_data(data, nside=nside, RACol=RACol,
                           DecCol=DecCol, filterCol=filterCol)
    print('booooo', df_pix)
    # get matching pixels<-> FP pos
    df_fp.set_display_mode()  # mandatory if plot_fp_pixels is to be used
    pix_obs = df_fp.pix_to_obs(df_pix)
    df_fp.plot_fp_pixels(pixels=df_pix, signal=pix_obs)

    print('processing', j, time.time()-time_ref)
    if output_q is not None:
        return output_q.put({j: pix_obs})
    else:
        return pix_obs


def process_pixels(pixels, params, j=0, output_q=None):
    """
    Function to process pixels 

    Parameters
    ----------
    pixels : pandas df
        List of pixels to process.
    params : dict
        Parameters.
    j : int, optional
        flag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Queue for multiprocessing. The default is None.

    Returns
    -------
    pandas df
        pixels <-> observations.

    """

    data = params['data']
    df_fp = params['df_fp']
    RACol = params['RACol']
    DecCol = params['DecCol']
    filterCol = params['filterCol']

    pix_obs = pd.DataFrame()
    for i, pix in pixels.iterrows():
        dd = get_xy_pixels(data, pix['healpixID'],
                           pix['pixRA'],
                           pix['pixDec'],
                           RACol=RACol, DecCol=DecCol, filterCol=filterCol)
        pixl_obs = df_fp.pix_to_obs(dd)
        pix_obs = pd.concat((pix_obs, pixl_obs))

    if output_q is not None:
        return output_q.put({j: pix_obs})
    else:
        return pix_obs


def plot_nvisits(nside, tab, minx, maxx, leg='', xval='count'):
    """
    Method to plot nvisits (Mollview)

    Parameters
    ----------
    nside : int
        healpix nside parameter.
    tab : pandas df
        Data to plot.
    minx : float
        min x-axis.
    maxx : float
        max x-axis.
    leg : str, optional
        Legend. The default is ''.
    xval : str, optional
        column val. The default is 'count'.

    Returns
    -------
    None.

    """

    import numpy as np
    npix = hp.nside2npix(nside=nside)
    norm = plt.cm.colors.Normalize(minx, maxx)
    cmap = plt.cm.jet

    cmap.set_under('w')

    hpID = 'healpixID'
    hpxmap = np.zeros(npix, dtype=float)
    r = []
    for healpixID in np.unique(tab[hpID]):
        ii = tab[hpID] == healpixID
        sel = tab[ii]
        r.append((healpixID, np.median(sel[xval])))
    rt = np.rec.fromrecords(r, names=[hpID, xval])
    hpxmap[rt[hpID].astype(int)] = rt[xval]

    hp.mollview(hpxmap, min=minx, max=maxx, cmap=cmap,
                title=leg, nest=True, norm=norm)

    hp.graticule()


parser = OptionParser(description='LSST FP test')

parser.add_option('--dbDir', type='str',
                  default='../DB_Files', help='db dir [%default')
parser.add_option('--dbName', type='str',
                  default='baseline_v3.6_10yrs', help='db name [%default]')
parser.add_option('--RACol', type='str', default='RA',
                  help="RA colname [%default]")
parser.add_option('--DecCol', type='str', default='Dec',
                  help="Dec colname [%default]")
parser.add_option('--filterCol', type='str', default='band',
                  help="filter colname [%default]")
parser.add_option('--fp_level', type='str', default='raft',
                  help="FP level [raft/ccd/sensor] [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = '{}.npy'.format(opts.dbName)
RACol = opts.RACol
DecCol = opts.DecCol
filterCol = opts.filterCol
fp_level = opts.fp_level

# FP instance
df_fp = FocalPlane(level=fp_level)
# quick check
# df_fp.check_fp(top_level='raft', low_level='ccd')

# load the data to process
data = get_simuData(dbDir, dbName)
data = pd.DataFrame(data)
nside = 128
print('data', len(data))
"""
# first method: use pixels as ref
# get (RA,Dec) window for these data
print(data.columns)
RA_min, RA_max, Dec_min, Dec_max = get_window(data, RACol=RACol,
                                              DecCol=DecCol,
                                              radius=np.sqrt(20./3.14))

# get pixels in this window

pixels = get_pixels_in_window(nside, RA_min, RA_max, Dec_min, Dec_max)

print('number of pixels', len(pixels))
# process these pixels
params = {}
params['data'] = data
params['df_fp'] = df_fp
params['RACol'] = RACol
params['DecCol'] = DecCol
params['filterCol'] = 'band'
res = multiproc(pixels, params, process_pixels, 8)


obsCol = 'observationId'
obsids = res[obsCol].to_list()

idx = data[obsCol].isin(obsids)
sel_data = data[idx][[obsCol, RACol, DecCol]]

mm = res.merge(sel_data, left_on=[obsCol],
               right_on=[obsCol], suffixes=['', ''])

print(mm, len(mm[obsCol].unique()))

tt = mm.healpixID.value_counts().to_frame(
    'count').rename_axis('healpixID').reset_index()

print(tt)

minx = np.min(tt['count'])
maxx = np.max(tt['count'])
plot_nvisits(nside, tt, minx, maxx)
fig, ax = plt.subplots()
ax.plot(mm[RACol], mm[DecCol], 'k.')
ax.plot(mm['pixRA'], mm['pixDec'], 'r*')
"""

params = {}
params['nside'] = nside
params['fp_level'] = fp_level
params['RACol'] = RACol
params['DecCol'] = DecCol
params['filterCol'] = filterCol
# get proj pixels for obs
time_ref = time.time()
# pix_obs = multiproc(data, params, process_pointings, 1)
pix_obs = process_pointings(data[:1], params)

plt.show()


"""
print(test)

# load data to process
df_fp = FocalPlane(level='ccd')

df_fp.check_fp(top_level='raft', low_level='ccd')


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


print(pixels)

print(test)

nside = 64
params = {}
params['nside'] = 128
params['fp_level'] = 'ccd'
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
"""
