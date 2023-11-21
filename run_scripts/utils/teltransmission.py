from sn_telmodel.sn_telescope import Telescope
from sn_telmodel import plt
import os


def plot_throughputs(tela, fig=None, ax=None, ls='solid', label=''):

    if fig is None:
        fig, ax = plt.subplots()

    for i, band in enumerate(tela.filterlist):
        labelb = None
        if i == 0:
            labelb = label
        ax.plot(tela.lsst_atmos_aerosol[band].wavelen,
                tela.lsst_atmos_aerosol[band].sb,
                linestyle=ls, color=tela.filtercolors[band],
                label=labelb)


def get_telescope(through_dir='throughputs', tag='1.8'):

    path = os.getcwd()
    cmd = 'cd {}'.format(through_dir)

    os.chdir(through_dir)
    cmd = 'git checkout tags/{}'.format(tag)
    os.system(cmd)
    os.chdir(path)

    airmass = 1.2
    tela = Telescope(
        airmass=airmass, through_dir='throughputs/baseline',
        atmos_dir='throughputs/atmos')

    return tela
# telescope.Plot_Throughputs()


tela = get_telescope(tag='1.5')
telb = get_telescope(tag='1.9')


fig, ax = plt.subplots(figsize=(12, 8))

plot_throughputs(tela, fig, ax, label='1.5', ls='dashed')

plot_throughputs(telb, fig, ax, label='1.9', ls='solid')

# plot_throughputs(telc, fig, ax, label='new', ls='dotted')

ax.grid()

ax.legend(fancybox=False, numpoints=1)

ax.set_xlabel('Wavelength (nm)', fontweight='bold')
ax.set_ylabel('Sb (0-1)')
fig.suptitle('System throughput')
ax.set_ylim([0, None])
plt.show()
