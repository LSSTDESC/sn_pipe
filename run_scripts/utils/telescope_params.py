import numpy as np
from sn_telmodel.sn_telescope import get_telescope
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
opts, args = parser.parse_args()

# config = dict(zip(['tag','label'],[['1.5','1.9'],['Al_Ag_Al','Ag_Ag_Ag']]))

tel_dir = opts.tel_dir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
airmass = opts.airmass
tag = opts.tag
aerosol = 1

telb = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(telb, throughputsDir)
atmos_dir = '{}/{}'.format(telb, atmosDir)
telescope = get_telescope(tel_dir=telb,
                          through_dir=through_dir,
                          atmos_dir=atmos_dir,
                          tag=tag, load_components=True,
                          airmass=airmass, aerosol=aerosol)


bands = 'ugrizy'

exptime = 30
plateScale = 0.2  # pixel size ''
df = pd.DataFrame(['u', 'g', 'r', 'i', 'z', 'y'], columns=['band'])
zp = dict(zip(bands, [telescope.zp(b) for b in bands]))
mag_sky = dict(zip(bands, [telescope.mag_sky(b) for b in bands]))
flux_sky = dict(zip(bands, [telescope.flux_sky(b, exptime) for b in bands]))
m5 = dict(zip(bands, [telescope.m5(b, exptime) for b in bands]))

df['zp'] = [telescope.zp(b) for b in bands]
df['flux_zp'] = [telescope.counts_zp(b) for b in bands]
df['msky'] = [telescope.mag_sky(b) for b in bands]
df['flux_sky'] = [telescope.flux_sky(b, exptime) for b in bands]
df['flux_sky_mag'] = 10**(-0.4*(df['msky']-df['zp']))*plateScale**2
df['FWHMeff'] = [telescope.FWHMeff(b) for b in bands]
df['m5'] = [telescope.m5(b, exptime) for b in bands]


df = df.rename(columns={"zp": "zp (AB)",
                        "flux_zp": "flux_zp (pe/s/pix)",
                        "flux_sky": "flux_sky (pe/s/pix)",
                        "flux_sky_mag": "flux_sky_mag (pe/s/pix)",
                        "FWHMeff": "FWHMEff ('')",
                        "m5": "m5 (exptime: {} s)".format(exptime),
                        "msky": "msky (/\"2)"})
df = df.round(2)
pd.set_option('display.colheader_justify', 'center')
print(df.to_string(index=False))
