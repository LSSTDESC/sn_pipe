import numpy as np
from sn_telmodel.sn_telescope import Telescope
import pandas as pd
telescope = Telescope(airmass=1.2)

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
