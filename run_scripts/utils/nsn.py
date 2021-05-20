from sn_tools.sn_rate import NSN
from optparse import OptionParser
import numpy as np

parser = OptionParser()

parser.add_option("--H0", type=float, default=70.,
                  help="H0 parameter[%default]")
parser.add_option("--Om0", type=float, default=0.3,
                  help="Omega0 parameter[%default]")
parser.add_option("--min_rf_phase", type=float, default=-15.,
                  help="min rf phase[%default]")
parser.add_option("--max_rf_phase", type=float, default=30.,
                  help="max rf phase[%default]")
parser.add_option("--season_length", type=float, default=180.,
                  help="season length[%default]")
parser.add_option("--zmin", type=float, default=0.01,
                  help="zmin [%default]")
parser.add_option("--zmax", type=float, default=1.0,
                  help="zmax [%default]")
parser.add_option("--dz", type=float, default=0.01,
                  help="dz for z binning [%default]")
parser.add_option("--survey_area", type=float, default=9.6,
                  help="survey area in deg2 [%default]")
parser.add_option("--scale_factor", type=float, default=1.,
                  help="scale factor to apply to nsn [%default]")


opts, args = parser.parse_args()

H0 = opts.H0
Om0 = opts.Om0
min_rf_phase = opts.min_rf_phase
max_rf_phase = opts.max_rf_phase
season_length = opts.season_length
zmin = opts.zmin
zmax = opts.zmax
dz = opts.dz
survey_area = opts.survey_area
scale_factor = opts.scale_factor

nsn_proc = NSN(H0, Om0, min_rf_phase, max_rf_phase)
nsn_tot = nsn_proc(zmin, zmax, dz, season_length, survey_area, scale_factor)

print('Number of supernovae', nsn_tot)
