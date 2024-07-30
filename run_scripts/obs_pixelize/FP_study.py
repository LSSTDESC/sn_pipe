from sn_tools.sn_obs import LSSTPointing_circular
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from descartes.patch import PolygonPatch
from shapely import affinity
import pandas as pd


def focal_plane(nx=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 8*15])),
                ny=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 2*15])),
                FoV=9.62,
                level='raft'):

    fov_str = FoV*(np.pi/180.)**2  # LSST fov in sr
    theta = 2.*np.arcsin(np.sqrt(fov_str/(4.*np.pi)))
    fpscale = np.tan(theta)

    xmin, xmax = -fpscale, fpscale
    ymin, ymax = -fpscale, fpscale

    dx = xmax-xmin
    dy = ymax-ymin

    d_elem_x = dx/nx[level]
    d_elem_y = dy/ny[level]

    xvalues = np.arange(xmin, xmax, d_elem_x)
    yvalues = np.arange(ymin, ymax, d_elem_y)
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
    indexes = ['raft', 'ccd', 'sensor']
    idx_level = indexes.index(level)
    for i in range(idx_level+1):
        vv = indexes[i]
        d_elem_xx = dx/nx[vv]
        d_elem_yy = dy/ny[vv]
        df_fp = get_index(df_fp, xmin, ymax, d_elem_xx, d_elem_yy, vv)

    # remove extra cells
    idx = df_fp['xc'] > xmin
    idx &= df_fp['xc'] < xmax
    idx &= df_fp['yc'] > ymin
    idx &= df_fp['yc'] < ymax

    return pd.DataFrame(df_fp[idx])


def get_index(df_fp, xmin, ymax, d_elem_x, d_elem_y, level):

    ipos = (df_fp['xc']-xmin)/d_elem_x+1
    jpos = (ymax-df_fp['yc'])/d_elem_y+1
    lpj = '{}_j'.format(level)
    lpi = '{}_i'.format(level)
    df_fp[lpj] = ipos
    df_fp[lpi] = jpos
    df_fp[lpj] = df_fp[lpj].astype(int)
    df_fp[lpi] = df_fp[lpi].astype(int)
    df_fp[level] = df_fp[lpi].astype(str)+'_'+df_fp[lpj].astype(str)
    df_fp = df_fp.drop(columns=[lpi, lpj])

    return df_fp


def check_fp(df_fp, top_level='raft', low_level='ccd'):

    posi = '{}_i'.format(top_level)
    posj = '{}_j'.format(top_level)

    ijpos = df_fp[top_level].unique()

    print('Number of {}s'.format(top_level), len(ijpos))
    if low_level != top_level:
        for vv in ijpos:
            idx = df_fp[top_level] == vv
            sel = df_fp[idx]
            idxb = sel[low_level].unique()
            print('{}'.format(top_level), vv,
                  'N {}s'.format(low_level), len(idxb))


def remove_ccd(df_fp, ccdlist):

    idx = df_fp['ccd'].isin(ccdlist)

    return pd.DataFrame(df_fp[~idx])


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

xvalues = np.arange(xmin, xmax, d_elem)
yvalues = np.arange(ymin, ymax, d_elem)
xv, yv = np.meshgrid(xvalues, yvalues)

print(xv.shape)

df_fp = pd.DataFrame(xv.flatten(), columns=['x'])
df_fp['y'] = yv.flatten()
df_fp['xc'] = df_fp['x']+0.5*d_elem
df_fp['yc'] = df_fp['y']+0.5*d_elem
df_fp['xmin'] = df_fp['xc']-0.5*d_elem
df_fp['xmax'] = df_fp['xc']+0.5*d_elem
df_fp['ymin'] = df_fp['yc']-0.5*d_elem
df_fp['ymax'] = df_fp['yc']+0.5*d_elem

df_fp = focal_plane(level='sensor')
print('allo', d_elem, len(df_fp))
print(df_fp)

ccd = {}
ccd['to_remove'] = ['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                    '1_13', '1_14', '1_15', '2_14', '2_15', '3_15',
                    '15_1', '15_2', '15_3', '14_1', '14_2', '13_1',
                    '15_13', '15_14', '15_15', '14_14', '14_15', '13_15']
ccd['guide'] = ['2_3', '3_2', '2_13', '3_14',
                '14_3', '13_2', '14_13', '13_14']
ccd['wave'] = ['3_3', '3_13', '13_3', '13_13']

df_fp = remove_ccd(df_fp, ccd['to_remove'])
df_fp = remove_ccd(df_fp, ccd['guide'])
df_fp = remove_ccd(df_fp, ccd['wave'])
check_fp(df_fp, top_level='sensor', low_level='sensor')

fig, ax = plt.subplots()
ax.plot(df_fp['x'], df_fp['y'], 'k.')
ax.plot(df_fp['xc'], df_fp['yc'], 'r*')

ax.plot(df_fp['xmin'], df_fp['ymin'], color='b',
        marker='s', mfc='None', linestyle='None')
ax.plot(df_fp['xmin'], df_fp['ymax'], color='b',
        marker='s', mfc='None', linestyle='None')
ax.plot(df_fp['xmax'], df_fp['ymin'], color='b',
        marker='s', mfc='None', linestyle='None')
ax.plot(df_fp['xmax'], df_fp['ymax'], color='b',
        marker='s', mfc='None', linestyle='None')

plt.show()

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
