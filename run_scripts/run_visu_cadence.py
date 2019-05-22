from sn_tools.sn_visu import CadenceMovie, SnapNight
from optparse import OptionParser
import matplotlib.pyplot as plt

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--nights", type="int", default='2',
                  help="number of nights to display [%default]")
parser.add_option("--saveMovie", type="int", default=0,
                  help="flag to save movie [%default]")
parser.add_option("--realTime", type="int", default=0,
                  help="real-time mode [%default]")
parser.add_option("--saveFig", type="int", default=0,
                  help="save fig end of the night [%default]")
parser.add_option("--areaTime", type="int", default=0,
                  help="display observed area [%default]")
parser.add_option("--dispType", type="str", default='cadence',
                  help="what to display[%default]")
opts, args = parser.parse_args()

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
dbName = opts.dbName
nights = opts.nights
saveMovie = opts.saveMovie
realTime = opts.realTime
saveFig = opts.saveFig
areaTime = opts.areaTime
dispType = opts.dispType


if dispType == 'cadence':
    CadenceMovie(dbDir=dbDir, dbName=dbName, title=dbName, nights=nights,
                 saveMovie=saveMovie, realTime=realTime, saveFig=saveFig, areaTime=areaTime)

if dispType == 'snapshot':
    SnapNight(dbDir=dbDir, dbName=dbName, saveFig=saveFig,
              nights=nights, realTime=realTime, areaTime=areaTime)

