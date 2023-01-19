import numpy as np
from scipy.optimize import curve_fit
from sn_plotter_metrics import plt


def plot(res, yvar='zp', yleg='zp [mag]', save_fit=False):
    """
    Function to plot result and fit (linear)

    Parameters
    ----------
    res : array
        data to process.
    yvar : str, optional
        y-axis var. The default is 'zp'.
    yleg : str, optional
        y-axis legend. The default is 'zp [mag]'.
    save_fit : bool, optional
        to save fit parameters. The default is False.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))

    r = []
    for b in 'ugrizy':
        idx = res['band'] == b
        sel = res[idx]
        xdata = sel['airmass']
        ydata = sel[yvar]
        ax.plot(xdata, ydata, 'k.')
        popt, pcov = curve_fit(func, xdata, ydata)
        print(b, popt)
        ax.plot(xdata, func(xdata, *popt), 'r-')
        r.append((b, popt[0], popt[1]))

    ax.set_xlabel('airmass')
    ax.set_ylabel(yleg)
    ax.grid()

    print(r)
    if save_fit:
        res = np.rec.fromrecords(r, names=['band', 'coeff', 'zp'])
        np.save('{}_airmass.npy'.format(yvar), res)


def func(x, a, b):

    return a*x+b


res = np.load('zero_points_airmass.npy')

plot(res, save_fit=True)
plot(res, yvar='zp_adu_sec', yleg='zp-> ADU/s')
plt.show()
