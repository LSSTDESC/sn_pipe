import sys
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
import sn_metrics_input
import sn_script_input
import os
from sn_tools.sn_process import Process


def add_parser(parser, confDict):
    for key, vals in confDict.items():
        vv = vals[1]
        if vals[0] != 'str':
            vv = eval('{}({})'.format(vals[0], vals[1]))
        parser.add_option('--{}'.format(key), help='{} [%default]'.format(
            vals[2]), default=vv, type=vals[0], metavar='')


metric = sys.argv[1]

#print('metric', metric)

# get all possible simulation parameters and put in a dict
path_metric_input = sn_metrics_input.__path__
path_process_input = sn_script_input.__path__
confDict_gen = make_dict_from_config(
    path_process_input[0], 'config_process.txt')
confDict_metric = make_dict_from_config(
    path_metric_input[0], 'config_{}.txt'.format(metric))

parser = OptionParser()
# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict_gen)
add_parser(parser, confDict_metric)

opts, args = parser.parse_args()

# load the new values
metricDict = {}
metricProc = {}
for key, vals in confDict_metric.items():
    metricDict[key] = eval('opts.{}'.format(key))
for key, vals in confDict_gen.items():
    metricProc[key] = eval('opts.{}'.format(key))

metricDict['RAmin'] = opts.RAmin
metricDict['RAmax'] = opts.RAmax
metricDict['Decmin'] = opts.Decmin
metricDict['Decmax'] = opts.Decmax
metricDict['npixels'] = opts.npixels


# print(metricDict)

#print('Start processing...', opts)

# prepare outputDir
nodither = ''
if opts.remove_dithering:
    nodither = '_nodither'
outputDir = '{}/{}{}/{}'.format(opts.outDir,
                                opts.dbName, nodither, metric)

if opts.fieldType == 'DD':
    outputDir += '_{}'.format(opts.fieldName)

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

metricDict['outDir'] = outputDir
#metricDict['season'] = season_int
metricDict['metric'] = metric
metricDict['OSName'] = opts.dbName
metricDict['fieldType'] = opts.fieldType
metricDict['fieldName'] = opts.fieldName
metricDict['nside'] = opts.nside
classname = '{}MetricWrapper'.format(metric)
metricList = []

exec('from metricWrapper import {}MetricWrapper'.format(metric))
metricList.append(globals()[classname](**metricDict))

# print('seasons and metric', opts.seasons,
#      metric, opts.pixelmap_dir, opts.npixels)

metricProc['metricList'] = metricList
metricProc['outDir'] = outputDir

#print('processing', metricProc)
process = Process(**metricProc)
