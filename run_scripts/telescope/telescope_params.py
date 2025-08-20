import numpy as np
from sn_telmodel.sn_throughputs import get_telescope
import pandas as pd
from optparse import OptionParser

parser = OptionParser(description='Script to plot telescope throughputs')

parser.add_option('--tel_dir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.04,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=5.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone value [%default]')
parser.add_option('--gain', type=float, default=2.5,
                  help='electronic gain [%default]')
parser.add_option('--pressure', type=float, default=743.,
                  help='pressure on the Cerro Pachon [%default]')
parser.add_option('--atmos_type', type=str, default='obsatmo',
                  help='how is the atmos estimated (obsatmo, from_file) [%default]')
parser.add_option('--fwhmeff', type=str, default='0.92,0.87,0.83,0.80,0.78,0.76',
                  help='FWHMeff values for ugrizy bands [%default]')

opts, args = parser.parse_args()

# config = dict(zip(['tag','label'],[['1.5','1.9'],['Al_Ag_Al','Ag_Ag_Ag']]))

tel_dir = opts.tel_dir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
airmass = opts.airmass
tag = opts.tag
gain = opts.gain
aerosol = opts.aerosol
pwv = opts.pwv
ozone = opts.ozone
pressure = opts.pressure
atmos_type = opts.atmos_type
fwhmeff = opts.fwhmeff.split(',')
fwhmeff = list(map(float, fwhmeff))

telb = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(telb, throughputsDir)
atmos_dir = '{}/{}'.format(telb, atmosDir)
telescope = get_telescope(tel_dir=telb,
                          through_dir=through_dir,
                          atmos_dir=atmos_dir,
                          atmos_type=atmos_type,
                          tag=tag, load_components=True,
                          airmass=airmass, aerosol=aerosol,
                          pwv=pwv, ozone=ozone, gain=gain, pressure=pressure)


bands = 'ugrizy'

exptime = 30
nexp = 1
plateScale = 0.2  # pixel size ''
bands = 'ugrizy'

telescope.data['FWHMeff'] = dict(
    zip('ugrizy', fwhmeff))

telescope.etc(exptime, plateScale, nexp)


m5_from_file = [23.697, 24.973, 24.516, 24.127, 23.556, 22.550]
dict_m5_ref = dict(zip(bands, m5_from_file))

for b in bands:
    m5_res = telescope.data['m5'][b]
    diff_m5 = m5_res-dict_m5_ref[b]
    print(b, diff_m5, 10**(-0.4*diff_m5))
