from sn_plotter_simu.visuLC import VisuLC
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--metaFile', type=str, default='Meta_fit.hdf5',
                  help='meta file name to process [%default]')
parser.add_option('--metaDir', type=str, default='dataLC',
                  help='meta dir [%default]')
parser.add_option('--SNFile', type=str, default='None',
                  help='SN file[%default]')
parser.add_option('--SNDir', type=str, default='None',
                  help='SN dir [%default]')

opts, args = parser.parse_args()

opts_dict = vars(opts)

visu = VisuLC(**opts_dict)

while 1:
    answer = input('SN to plot? ')

    if answer != 'exit':
        visu.plot(answer)

    else:
        break
