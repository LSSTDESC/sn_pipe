# import matplotlib.pyplot as plt
#import matplotlib
# matplotlib.use('agg')
import numpy as np
from optparse import OptionParser
import time
import os
#import glob
import sys
#import random
#import pandas as pd
from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from metricWrapper import ObsRateMetricWrapper, NSNMetricWrapper
from metricWrapper import NSNYMetricWrapper
from metricWrapper import SaturationMetricWrapper
from metricWrapper import SLMetricWrapper
from metricWrapper import SNRTimeMetricWrapper
from sn_tools.sn_process import Process

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str",
                  default='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='MetricOutput',
                  help="output dir [%default]")
parser.add_option("--templateDir", type="str", default='/sps/lsst/data/dev/pgris/Templates_final_new',
                  help="template dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type DD or WFD[%default]")
parser.add_option("--zmax", type="float", default='1.2',
                  help="zmax for simu [%default]")
parser.add_option("--remove_dithering", type="int", default='0',
                  help="remove dithering for DDF [%default]")
parser.add_option("--simuType", type="int", default='0',
                  help="flag for new simulations [%default]")
parser.add_option("--saveData", type="int", default='0',
                  help="flag to dump data on disk [%default]")
parser.add_option("--metric", type="str", default='cadence',
                  help="metric to process [%default]")
parser.add_option("--coadd", type="int", default='1',
                  help="nightly coaddition [%default]")
# parser.add_option("--nodither", type="str", default='',
#                  help="to remove dithering - for DDF only[%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - for WDF only[%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - for WDF only[%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - for WDF only[%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - for WDF only[%default]")
parser.add_option("--proxy_level", type=int, default=2,
                  help="proxy level for the metric[%default]")
parser.add_option("--T0s", type=str, default='all',
                  help="T0 values to consider[%default]")
parser.add_option("--lightOutput", type=int, default=0,
                  help="light LC output[%default]")
parser.add_option("--outputType", type=str, default='zlims',
                  help="outputType of the metric[%default]")
parser.add_option("--seasons", type=str, default='-1',
                  help="seasons to process[%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for the metric[%default]")
parser.add_option("--timer", type=int, default=0,
                  help="timer mode for the metric[%default]")
parser.add_option("--ploteffi", type=int, default=0,
                  help="plot efficiencies for the metric[%default]")
parser.add_option("--z", type=float, default=0.2,
                  help="redshift for the metric[%default]")
parser.add_option("--band", type=str, default='r',
                  help="band for the metric[%default]")
parser.add_option("--dirRefs", type=str, default='reference_files',
                  help="dir of reference files for the metric[%default]")
parser.add_option("--dirFake", type=str, default='input/Fake_cadence',
                  help="dir of fake files for the metric[%default]")
parser.add_option("--names_ref", type=str, default='SNCosmo',
                  help="ref name for the ref files for the metric[%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="Supernova stretch[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="Supernova color[%default]")
parser.add_option("--pixelmap_dir", type=str, default='',
                  help="dir where to find pixel maps[%default]")
parser.add_option("--npixels", type=int, default=0,
                  help="number of pixels to process[%default]")
parser.add_option("--nclusters", type=int, default=0,
                  help="number of clusters in data (DD only)[%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius around clusters (DD and Fakes)[%default]")
parser.add_option("--ebvofMW", type=float, default=-1.0,
                  help="E(B-V) of MW for dust corrections[%default]")
parser.add_option("--fieldName", type=str, default='COSMOS',
                  help="fieldName - for DD only[%default]")
parser.add_option("--healpixIDs", type=str, default='',
                  help="list of healpixIds to process [%default]")


opts, args = parser.parse_args()

print('Start processing...', opts)


# prepare outputDir
nodither = ''
if opts.remove_dithering:
    nodither = '_nodither'
outputDir = '{}/{}{}/{}'.format(opts.outDir,
                                opts.dbName, nodither, opts.metric)

healpixIDs = []
if opts.healpixIDs != '':
    healpixIDs = list(map(int, opts.healpixIDs.split(',')))

if opts.fieldType == 'DD':
    outputDir += '_{}'.format(opts.fieldName)

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

# List of (instance of) metrics to process
metricList = []

# check whether the metric is available

available_metrics = ['NSN', 'NSNY', 'Cadence', 'SL',
                     'ObsRate', 'SNRr', 'SNRz', 'Saturation', 'SNRTime']
if opts.metric not in available_metrics:
    print('Sorry to inform you that', opts.metric, 'is not a metric available')
    print('list of possible metrics:')
    print(available_metrics)
    sys.exit(0)

season_int = list(opts.seasons.split(','))
if season_int[0] == '-':
    season_int = -1
else:
    season_int = list(map(int, season_int))

metricname = opts.metric
if 'SNR' in opts.metric and 'SNRTime' not in metricname:
    metricname = 'SNR'

classname = '{}MetricWrapper'.format(metricname)


metricList.append(globals()[classname](name=opts.metric, season=season_int,
                                       coadd=opts.coadd, fieldType=opts.fieldType,
                                       nside=opts.nside,
                                       RAmin=opts.RAmin, RAmax=opts.RAmax,
                                       Decmin=opts.Decmin, Decmax=opts.Decmax,
                                       npixels=opts.npixels, metadata=opts, outDir=outputDir, ebvofMW=opts.ebvofMW))

print('seasons and metric', season_int,
      metricname, opts.pixelmap_dir, opts.npixels)


process = Process(opts.dbDir, opts.dbName, opts.dbExtens,
                  opts.fieldType, opts.fieldName, opts.nside,
                  opts.RAmin, opts.RAmax,
                  opts.Decmin, opts.Decmax,
                  opts.saveData, opts.remove_dithering,
                  outputDir, opts.nproc, metricList,
                  opts.pixelmap_dir, opts.npixels,
                  opts.nclusters, opts.radius, healpixIDs)
