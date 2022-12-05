import sys
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
import sn_metrics_input
import os
from sn_tools.sn_process import Process_new


def add_parser(parser, confDict):
    for key, vals in confDict.items():
        vv = vals[1]
        if vals[0] != 'str':
            vv = eval('{}({})'.format(vals[0], vals[1]))
        parser.add_option('--{}'.format(key), help='{} [%default]'.format(
            vals[2]), default=vv, type=vals[0], metavar='')


metric = sys.argv[1]

print('metric', metric)

# get all possible simulation parameters and put in a dict
path_metric_input = sn_metrics_input.__path__
confDict_gen = make_dict_from_config(path_metric_input[0], 'config_metric.txt')
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

print(metricDict)

print('Start processing...', opts)

# prepare outputDir
nodither = ''
if opts.remove_dithering:
    nodither = '_nodither'
outputDir = '{}/{}{}/{}'.format(opts.outDir,
                                opts.dbName, nodither, metric)

healpixIDs = []

if opts.healpixIDs.strip() != '\'\'':
    healpixIDs = list(map(int, opts.healpixIDs.split(',')))

if opts.fieldType == 'DD':
    outputDir += '_{}'.format(opts.fieldName)

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

metricDict['outDir'] = outputDir
#metricDict['season'] = season_int
metricDict['metric'] = metric
metricDict['OSName'] = opts.dbName
classname = '{}MetricWrapper'.format(metric)
metricList = []

exec('from metricWrapper import {}MetricWrapper'.format(metric))
metricList.append(globals()[classname](**metricDict))

print('seasons and metric', opts.seasons,
      metric, opts.pixelmap_dir, opts.npixels)

metricProc['fieldType'] = opts.fieldType
metricProc['metricList'] = metricList
metricProc['fieldName'] = opts.fieldName
metricProc['outDir'] = opts.outDir
metricProc['healpixIDs'] = healpixIDs

print('processing', metricProc)
process = Process_new(**metricProc)


"""
process = Process_new(opts.dbDir, opts.dbName, opts.dbExtens,
                  opts.fieldType, opts.fieldName, opts.nside,
                  opts.RAmin, opts.RAmax,
                  opts.Decmin, opts.Decmax,
                  opts.saveData, opts.remove_dithering,
                  outputDir, opts.nproc, metricList,
                  opts.pixelmap_dir, opts.npixels,
                  opts.nclusters, opts.radius, healpixIDs)
"""
