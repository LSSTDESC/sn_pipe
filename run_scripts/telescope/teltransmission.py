from sn_telmodel.sn_telescope import get_telescope
from sn_telmodel import plt
import os
from optparse import OptionParser
import numpy as np


def plot_throughputs(tela, what, fig=None, ax=None,
                     ls='solid', label='', lw=3):
    """
    Function to plot throughputs

    Parameters
    ----------
    tela : Telescope class
        Telescope to draw.
    what : Bandpass
        bandpass to draw.
    fig : matplotlib Figure, optional
        Figure where the plot will be drawn. The default is None.
    ax : matplotlib axis, optional
        axis where the plot will be drawn. The default is None.
    ls : str, optional
        Line style. The default is 'solid'.
    label : str, optional
        Label. The default is ''.
    lw : int, optional
        Line width. The default is 3.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots()

    for i, band in enumerate(tela.filterlist):
        labelb = None
        if i == 0:
            labelb = label
            ax.plot(what[band].wavelen,
                    what[band].sb,
                    linestyle=ls, color='k',
                    label=labelb, lw=lw)

        ax.plot(what[band].wavelen,
                what[band].sb,
                linestyle=ls, color=tela.filtercolors[band],
                label=None, lw=lw)


def plot_optical(tel, fig=None, ax=None, ls='solid', label='',
                 vars=['lens1', 'lens2', 'lens3', 'm1', 'm2', 'm3'], lw=3):
    """
    Function to plot lenses and mirrors throughputs

    Parameters
    ----------
    tela : Telescope class
        Telescope to draw.
    what : Bandpass
        bandpass to draw.
    fig : matplotlib Figure, optional
        Figure where the plot will be drawn. The default is None.
    ax : matplotlib axis, optional
        axis where the plot will be drawn. The default is None.
    ls : str, optional
        Line style. The default is 'solid'.
    label : str, optional
        Label. The default is ''.
    vars : list(str), optional
        Variables to draw.
        The default is ['lens1', 'lens2', 'lens3', 'm1', 'm2', 'm3']
    lw : int, optional
        Line width. The default is 3.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots(ncols=2, nrows=3)

    ipos = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    pos = dict(zip(vars, ipos))

    import numpy as np
    for vv in vars:
        axb = ax[pos[vv]]
        telb = tel.optical[vv]
        idx = telb.wavelen >= 860
        idx &= telb.wavelen < 861
        print(vv, np.mean(telb.wavelen[idx]), np.mean(telb.sb[idx]))

        if vv != 'lens1':
            axb.plot(telb.wavelen,
                     telb.sb,
                     linestyle=ls, color='k',
                     label=None, lw=lw)
        else:
            axb.plot(telb.wavelen,
                     telb.sb,
                     linestyle=ls, color='k',
                     label=label, lw=lw)
        axb.text(0.3, 0.1, vv, color='dimgrey',
                 fontsize=15,
                 transform=axb.transAxes, ha='right')

        axb.set_xlabel('Wavelength (nm)', fontweight='bold')
        axb.set_ylim([0, 1.05])

        if pos[vv][1] == 1:
            axb.tick_params(axis='y', colors='white')
            axb.set_yticklabels([])

        if pos[vv] == (1, 0):
            axb.set_ylabel('Throughput (0-1)')

        if vv == 'lens1':
            axb.legend(bbox_to_anchor=(
                0.5, 0.98), ncol=2, fontsize=15, frameon=False)
        axb.grid(visible=True)


def plot_single(tel, fig=None, ax=None,
                ls='solid', label='', var='detector', lw=3):
    """
    Function to plot a single parameter.

    Parameters
    ----------
    tela : Telescope class
        Telescope to draw.
    what : Bandpass
        bandpass to draw.
    fig : matplotlib Figure, optional
        Figure where the plot will be drawn. The default is None.
    ax : matplotlib axis, optional
        axis where the plot will be drawn. The default is None.
    ls : str, optional
        Line style. The default is 'solid'.
    label : str, optional
        Label. The default is ''.
    vars : str, optional
        Variables to draw.
        The default is 'dector'
    lw : int, optional
        Line width. The default is 3.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(tel.optical[var].wavelen,
            tel.optical[var].sb,
            linestyle=ls, color='k',
            label=label, lw=lw)


parser = OptionParser(description='Script to plot telescope throughputs')

parser.add_option('--tel_dir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='atmos_new',
                  help='atmosphere location dir [%default]')
parser.add_option('--tags', type=str, default='1.9,1.5',
                  help='tag versions of the throughputs [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.0,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone value [%default]')
opts, args = parser.parse_args()

# config = dict(zip(['tag','label'],[['1.5','1.9'],['Al_Ag_Al','Ag_Ag_Ag']]))

tel_dir = opts.tel_dir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
airmass = opts.airmass
pwv = opts.pwv
aerosol = opts.aerosol
ozone = opts.ozone

tags = opts.tags.split(',')


ls = dict(zip(tags, ['solid', 'dotted']))


tel = {}

for tag in tags:
    telb = '{}_{}'.format(tel_dir, tag)
    through_dir = '{}/{}'.format(telb, throughputsDir)
    atmos_dir = '{}/{}'.format(telb, atmosDir)
    tel[tag] = get_telescope(tel_dir=telb,
                             through_dir=through_dir,
                             atmos_dir=atmos_dir,
                             tag=tag, airmass=airmass,
                             aerosol=aerosol, pwv=pwv, oz=ozone)


fig, ax = plt.subplots(figsize=(12, 8))
figb, axb = plt.subplots(figsize=(12, 8), ncols=2, nrows=3)
figc, axc = plt.subplots(figsize=(12, 8))
figb.subplots_adjust(wspace=0., hspace=0.05)
figd, axd = plt.subplots(figsize=(12, 8))

for tag in tags:
    lab = 'v{}'.format(tag)
    plot_throughputs(tel[tag], tel[tag].lsst_atmos_aerosol,
                     fig, ax, label=lab, ls=ls[tag])
    plot_throughputs(tel[tag], tel[tag].filter,
                     figc, axc, label=lab, ls=ls[tag])
    plot_optical(tel[tag], figb, axb, label=lab, ls=ls[tag])
    plot_single(tel[tag], figd, axd, label=lab, ls=ls[tag])

# first plot
ax.grid()
ax.legend(bbox_to_anchor=(0.6, 1.08), ncol=2, fontsize=15, frameon=False)
ax.set_xlabel('Wavelength (nm)', fontweight='bold')
ax.set_ylabel('Throughput(0-1)')
tit = 'Telescope+airmass ({})+aerosol'.format(np.round(airmass, 1))
if airmass <= 0.1:
    tit = 'Telescope'
fig.suptitle(tit)
ax.set_ylim([0, None])

# second plot
axc.grid()
axc.legend(bbox_to_anchor=(0.6, 1.08), ncol=2, fontsize=15, frameon=False)
axc.set_xlabel('Wavelength (nm)', fontweight='bold')
axc.set_ylabel('Throughput (0-1)')
figc.suptitle('Filters')
axc.set_ylim([0, None])


# third plot
figb.suptitle('Lenses and Mirrors')

# fourth plot
axd.grid()
axd.legend(bbox_to_anchor=(0.6, 1.08), ncol=2, fontsize=15, frameon=False)
axd.set_xlabel('Wavelength (nm)', fontweight='bold')
axd.set_ylabel('Throughput (0-1)')
figd.suptitle('Detector (QE+coating)')
axd.set_ylim([0, None])


plt.show()
