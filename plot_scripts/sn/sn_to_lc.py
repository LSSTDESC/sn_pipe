from optparse import OptionParser

parser = OptionParser()

parser.add_option('--LCDirInput', type=str, default='dataLC',
                  help='meta dir [%default]')
parser.add_option('--SNFileInput', type=str, default='None',
                  help='SN file[%default]')
parser.add_option('--SNDirInput', type=str, default='None',
                  help='SN dir [%default]')

opts, args = parser.parse_args()

opts_dict = vars(opts)
