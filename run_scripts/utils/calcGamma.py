from sn_tools.sn_utils import Gamma
from sn_tools.sn_telescope import Telescope
import numpy as np

bands = 'grizy'
telescope = Telescope(airmass=1.2)

outName = 'gamma.hdf5'
mag_range = np.arange(13., 38., 0.05)
#exptimes = np.arange(0., 9000., 10.)
#exptimes[0] = 1.
#nexps = range(1, 200, 1)
nexps = range(1, 20, 1)
#nexps = [1, 2]
#single_exposure_time = [15., 30.]
single_exposure_time = range(1, 40, 1)

Gamma(bands, telescope, outName,
      mag_range=mag_range,
      single_exposure_time=single_exposure_time, nexps=nexps)
