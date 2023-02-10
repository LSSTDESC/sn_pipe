from sn_plotter_simu.visuLC import VisuLC
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--metaFileInput', type=str, default='Meta_fit.hdf5',
                  help='meta file name to process [%default]')
parser.add_option('--metaDirInput', type=str, default='dataLC',
                  help='meta dir [%default]')

opts, args = parser.parse_args()

metaFileInput = opts.metaFileInput
metaDirInput = opts.metaDirInput

visu = VisuLC(metaFileInput, metaDirInput)

while 1:
    answer = input('SN to plot? ')

    visu.plot(answer)
