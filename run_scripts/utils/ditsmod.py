from astropy.cosmology import w0waCDM
import matplotlib.pyplot as plt
import numpy as np

def distmod(H0, Omega_m, Omega_l, w0, wa,z):

    cosmology = w0waCDM(H0, Omega_m, Omega_l, w0, wa)
    lumidist = cosmology.luminosity_distance(z).value*1.e3
    distmod =  cosmology.distmod(z).value

    return distmod

H0 = 70.
Omega_m=0.3
Omega_l = 0.7
w0=-1.
wa=0.0

z = np.arange(0.2,1.1,0.01)

distmod_a = distmod(H0, Omega_m, Omega_l, w0, wa,z)

wa = 0.05

distmod_b = distmod(H0, Omega_m, Omega_l, w0, wa,z)

wa = 0.01

distmod_c = distmod(H0, Omega_m, Omega_l, w0, wa,z)


fig, ax = plt.subplots(nrows=2)

ax[0].plot(z,distmod_a)
ax[0].plot(z,distmod_b)
ax[0].set_yscale('log')
norm= distmod_a[-1]-distmod_b[-1]
ax[1].plot(z,(distmod_a-distmod_b)/distmod_a)
ax[1].plot(z,(distmod_a-distmod_c)/distmod_a)
ax[1].set_yscale('log')
plt.show()
