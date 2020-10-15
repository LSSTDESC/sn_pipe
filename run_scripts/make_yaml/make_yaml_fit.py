from optparse import OptionParser
import yaml
from importlib import import_module
import sn_fit_input as fit
import re
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse
from sn_tools.sn_io import decrypt_parser

path = fit.__path__

confDict = make_dict_from_config(path[0],'config_simulation.txt')

parser = OptionParser()
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')

parser.add_option('--fileName',help='output file name [%default]',default='config.yaml',type='str')

opts, args = parser.parse_args()

newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)


dd = make_dict_from_optparse(newDict)

print('config',dd)
with open(opts.fileName, 'w') as f:
    data = yaml.dump(dd, f)

