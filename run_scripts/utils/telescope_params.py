from sn_tools.sn_telescope import Telescope

telescope = Telescope(airmass=1.2)

bands = 'ugrizy'
zp = dict(zip(bands, [telescope.zp(b) for b in bands]))

print('zero points', zp)
print('mean wavelength', telescope.mean_wavelength)
