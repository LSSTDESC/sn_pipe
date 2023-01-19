import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b):

    return a*x+b


res = np.load('zero_points_airmass.npy')

fig, ax = plt.subplots(figsize=(12, 8))
for b in 'ugrizy':
    idx = res['band'] == b
    sel = res[idx]
    xdata = sel['airmass']
    ydata = sel['zp']
    ax.plot(xdata, ydata, 'k.')
    popt, pcov = curve_fit(func, xdata, ydata)
    print(b, popt)
    ax.plot(xdata, func(xdata, *popt), 'r-')

ax.set_xlabel('airmass')
ax.set_ylabel('zp [mag]')
plt.show()
