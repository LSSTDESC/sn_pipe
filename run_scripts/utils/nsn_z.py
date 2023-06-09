from sn_tools.sn_rate import SN_Rate
import matplotlib.pyplot as plt
import numpy as np

sn_rate = SN_Rate(Om0=0.3)

zmin = 0.01
zmax = 1.1
dz = 0.01
survey_area = 9.6
duration = 180.
account_for_edges = True

zz, rate, err_rate, nsn, err_nsn = sn_rate(
    zmin=zmin, zmax=zmax, dz=dz,
    account_for_edges=account_for_edges,
    duration=duration, survey_area=survey_area)

nsn_sum = np.cumsum(nsn)

fig, ax = plt.subplots()
ax.plot(zz, nsn_sum)
plt.show()
