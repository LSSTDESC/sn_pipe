import numpy as np
from sn_tools.sn_io import convert_save
import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='',
                  help="output dir [%default]")
parser.add_option("--metricName", type="str", default='SL',
                  help="metric name [%default]")
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type - WFD, DD, Fake [%default]")
parser.add_option("--remove_galactic", type="int", default=0,
                  help="remove galactic area [%default]")
parser.add_option("--unique", type="int", default=1,
                  help="to keep only uniques [%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/users/gris/MetricOutput'

dbName = opts.dbName
outDir = opts.outDir
if outDir == '':
    outDir = '/sps/lsst/users/gris/MetricSummary'

if not os.path.isdir(outDir):
    os.makedirs(outDir)

outputDir = '{}/{}'.format(outDir,dbName)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

metricName = opts.metricName
fieldType = opts.fieldType
remove_galactic = opts.remove_galactic
unique = opts.unique

inputDir = '{}/{}/{}'.format(dbDir,dbName,metricName)
convert_save(inputDir,dbName,metricName,outputDir,fieldType=fieldType,objtype='astropyTable',unique=unique,remove_galactic=remove_galactic)

