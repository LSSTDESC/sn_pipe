from sn_tools.sn_visu import CadenceMovie, SnapNight,MoviePixels
from optparse import OptionParser
import numpy as np


def multi_MoviePixels(dbDir,dbDir_pixels,figDir,movieDir,dbName, saveMovie, realTime,
                    saveFig,nightmin,nightmax,ffmpeg_cmd,mode,j, output_q):

    MoviePixels(dbDir=dbDir,dbDir_pixels=dbDir_pixels,
                figDir=figDir,movieDir=movieDir,
                dbName=dbName, saveMovie=saveMovie,
                realTime=realTime,
                saveFig=saveFig,
                nightmin=nightmin,nightmax=nightmax,
                ffmpeg_cmd=ffmpeg,mode=mode)

        
    if output_q is not None:
        return output_q.put({j:1})
    else:
        return 1


parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name(s) [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db', help="db dir [%default]")
parser.add_option("--dbDir_pixels", type="str", default='../ObsPixelized_circular_new/', help="db dir for pixel data[%default]")
parser.add_option("--figDir", type="str", default='../OS_Figures', help="dir for figures [%default]")
parser.add_option("--movieDir", type="str", default='../OS_Movies', help="dir for movies [%default]")
parser.add_option("--nights", type="str", default='1,2',
                  help="list of nights to display  [%default]")
parser.add_option("--saveMovie", type="int", default=0,
                  help="flag to save movie [%default]")
parser.add_option("--realTime", type="int", default=0,
                  help="real-time mode [%default]")
parser.add_option("--saveFig", type="int", default=0,
                  help="save fig end of the night [%default]")
parser.add_option("--areaTime", type="int", default=0,
                  help="display observed area [%default]")
parser.add_option("--dispType", type="str", default='cadence',
                  help="what to display (cadence,snapshot,moviepixels) [%default]")
parser.add_option("--ffmpeg", type="str", default='ffmpeg',
                  help="command to use ffmpeg [%default]")
parser.add_option("--mode", type="str", default='interactive',
                  help="mode to use MoviePixels (interactive/batch) [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbDir_pixels = opts.dbDir_pixels
figDir = opts.figDir
movieDir = opts.movieDir
dbName = opts.dbName
saveMovie = opts.saveMovie
realTime = opts.realTime
saveFig = opts.saveFig
areaTime = opts.areaTime
dispType = opts.dispType
ffmpeg = opts.ffmpeg
mode = opts.mode

if '-' not in opts.nights:
    nights = list(map(int,opts.nights.split(',')))
else:
    nights = list(map(int,opts.nights.split('-')))
    nights = range(np.min(nights),np.max(nights))

if dispType == 'cadence':
    CadenceMovie(dbDir=dbDir, dbName=dbName, title=dbName, nights=nights,
                 saveMovie=saveMovie, realTime=realTime, saveFig=saveFig, areaTime=areaTime)

if dispType == 'snapshot':
    SnapNight(dbDir=dbDir, dbName=dbName, saveFig=saveFig,
              nights=nights, realTime=realTime, areaTime=areaTime)

if dispType =='moviepixels':
    nightmin = np.min(nights)
    nightmax = np.max(nights)
    dbNames = opts.dbName.split(',')
    print('processing Movie Pixels',dbNames)
    if len(dbNames) < 2:
        MoviePixels(dbDir=dbDir,dbDir_pixels=dbDir_pixels,figDir=figDir,movieDir=movieDir,dbName=dbName, saveMovie=saveMovie, realTime=realTime,
                    saveFig=saveFig,nightmin=nightmin,nightmax=nightmax,ffmpeg_cmd=ffmpeg,mode=mode)
    else:
        import multiprocessing
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-{}'.format(j), target=multi_MoviePixels,
                                         args=(dbDir,dbDir_pixels,figDir,movieDir,dbName, saveMovie, realTime,
                                         saveFig,nightmin,nightmax,ffmpeg,mode,j,result_queue)) for j,dbName in enumerate(dbNames)]

        for p in procs:
            p.start()
        
        for p in multiprocessing.active_children():
            p.join()
