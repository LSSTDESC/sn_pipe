from sn_saturation.psf_pixels import PixelPSFSeeing
import time
from optparse import OptionParser


parser = OptionParser()
parser.add_option("--psf_type", type=str, default='single_gauss',
                  help="PSF (single_gauss/moffat)[%default]")
parser.add_option("--scanfast", type=int, default=0,
                  help="To fasten the processing (but lower precision)[%default]")

opts, args = parser.parse_args()
psf_type = opts.psf_type
scanfast = opts.scanfast


time_ref = time.time()
PixelPSFSeeing(psf_type)
