from optparse import OptionParser
import yaml
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_io import decrypt_parser

path = 'input/Fake_cadence'
confDict = make_dict_from_config(path, 'config_cadence.txt')

parser = OptionParser()
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

parser.add_option(
    '--fileName', help='output file name [%default]', default='config.yaml', type='str')

opts, args = parser.parse_args()

newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key] = (vals[0], newval)


dd = make_dict_from_optparse(newDict)

# few changes to be made here: transform some of the input to list
for vv in ['seasons', 'seasonLength']:
    what = dd[vv]
    if '-' not in what or what[0] == '-':
        nn = list(map(int, what.split(',')))
        print('ici', nn)
    else:
        nn = list(map(int, what.split('-')))
        nn = range(np.min(nn), np.max(nn))
    dd[vv] = nn

for vv in ['MJDmin']:
    what = dd[vv]
    if '-' not in what or what[0] == '-':
        nn = list(map(float, what.split(',')))
    else:
        nn = list(map(float, what.split('-')))
        nn = range(np.min(nn), np.max(nn))
    dd[vv] = nn

    # print('config',dd)
with open(opts.fileName, 'w') as f:
    data = yaml.safe_dump(dd, f)
# """
