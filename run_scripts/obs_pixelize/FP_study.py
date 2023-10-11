from sn_tools.sn_obs import LSSTPointing_circular
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from descartes.patch import PolygonPatch
from shapely import affinity

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

xmin = np.min(tt['y'])
xmax = np.max(tt['y'])

d = xmax-xmin
nx = 15
ny = int(nx/2)
ymin = 0.
d_elem = d/nx

rangey = range(-ny, ny+1)

num = 0
pixels = {}
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
#ax.plot(ttb['x'], ttb['y'], 'r*', linestyle='None')

for key, val in pixels.items():
    pf = PolygonPatch(val, facecolor=(0, 0, 0, 0), edgecolor='red')
    ax.add_patch(pf)


plt.show()
