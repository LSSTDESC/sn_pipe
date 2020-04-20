from sn_tools.sn_utils import Gamma
from sn_tools.sn_telescope import Telescope
import numpy as np

bands = 'grizy'
telescope = Telescope(airmass=1.2)

outName = 'gamma_test.hdf5'
mag_range = np.arange(15., 38., 1.)
exptimes = np.arange(1., 9000., 10.)
Gamma(bands, telescope, outName,
      mag_range=mag_range,
      exptimes=exptimes)
