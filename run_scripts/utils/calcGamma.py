from sn_tools.sn_utils import Gamma
from sn_tools.sn_telescope import Telescope
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--fieldType", type="str", default='WFD',
                  help="field Type (DD or WFD) [%default]")

opts, args = parser.parse_args()

fieldType = opts.fieldType

bands = 'grizy'
telescope = Telescope(airmass=1.2)

outName = 'gamma_{}.hdf5'.format(fieldType)
mag_range = np.arange(13., 38., 0.05)
#exptimes = np.arange(0., 9000., 10.)
#exptimes[0] = 1.
# DD parameters
if fieldType == 'DD':
    nexps = range(1, 250, 1)
    single_exposure_time = [15., 30.]
# WFD parameters
if fieldType == 'WFD':
    nexps = range(1, 20, 1)
    single_exposure_time = range(1, 40, 1)

#single_exposure_time = [15., 30.]
#nexps = [1, 2]


Gamma(bands, telescope, outName,
      mag_range=mag_range,
      single_exposure_time=single_exposure_time, nexps=nexps)
