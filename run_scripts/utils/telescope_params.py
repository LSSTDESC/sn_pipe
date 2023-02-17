from sn_telmodel.sn_telescope import Telescope
import pandas as pd
telescope = Telescope(airmass=1.2)

bands = 'ugrizy'

exptime = 600
df = pd.DataFrame(['u', 'g', 'r', 'i', 'z', 'y'], columns=['band'])
zp = dict(zip(bands, [telescope.zp(b) for b in bands]))
mag_sky = dict(zip(bands, [telescope.mag_sky(b) for b in bands]))
flux_sky = dict(zip(bands, [telescope.flux_sky(b, exptime) for b in bands]))
m5 = dict(zip(bands, [telescope.m5(b, exptime) for b in bands]))

df['zp (AB)'] = [telescope.zp(b) for b in bands]
df['msky'] = [telescope.mag_sky(b) for b in bands]
df['flux_sky'] = [telescope.flux_sky(b, exptime) for b in bands]
df['FWHMeff'] = [telescope.FWHMeff(b) for b in bands]
df['m5'] = [telescope.m5(b, exptime) for b in bands]


df = df.round(2)
print(df.to_string(index=False))
