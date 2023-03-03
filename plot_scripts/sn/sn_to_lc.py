from sn_plotter_simu.visuLC import SNToLC
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--metaDir', type=str, default='dataLC',
                  help='meta dir [%default]')
parser.add_option('--SNFile', type=str, default='None',
                  help='SN file[%default]')
parser.add_option('--SNDir', type=str, default='None',
                  help='SN dir [%default]')

opts, args = parser.parse_args()

opts_dict = vars(opts)

SNToLC(**opts_dict)
