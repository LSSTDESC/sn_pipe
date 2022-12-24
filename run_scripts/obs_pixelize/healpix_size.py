import healpy as hp
import numpy as np

lsst_camera_pixel = 0.2 #0.2 arcsec

for i in range(16):
    nside = 2**i
    pixArea = hp.nside2pixarea(nside, degrees=True)
    npixels = np.sqrt(pixArea)*3600/lsst_camera_pixel
    print(nside,pixArea,npixels)
