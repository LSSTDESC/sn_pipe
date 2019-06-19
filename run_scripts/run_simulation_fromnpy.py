import matplotlib
matplotlib.use('agg')
import numpy as np
import healpy as hp
#from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from sn_mafsim.sn_maf_simulation import SNMetric
from sn_tools.sn_obs import renameFields, pixelate, season, GetShape, ObsPixel
from optparse import OptionParser
import time
import multiprocessing
import yaml
import os


def makeYaml(input_file, dbDir, dbName, nside, nproc, num, diffflux,seasnum,outDir):

    with open(input_file, 'r') as file:
        filedata = file.read()

    prodid = '{}_seas_{}_{}'.format(dbName, seasnum, num)
    fullDbName = '{}/{}.npy'.format(dbDir, dbName)
    filedata = filedata.replace('prodid', prodid)
    filedata = filedata.replace('fullDbName', fullDbName)
    filedata = filedata.replace('nnproc', str(nproc))
    filedata = filedata.replace('nnside', str(nside))
    filedata = filedata.replace('outputDir', outDir)
    filedata = filedata.replace('diffflux', str(diffflux))
    filedata = filedata.replace('seasval', str(seasnum))
    return yaml.load(filedata, Loader=yaml.FullLoader)


def loop(healpixels, obsFocalPlane, dbDir, dbName, outDir, nside, diffflux,seasnum,x0_tab, j=0, output_q=None):

    config = makeYaml('input/param_simulation_gen.yaml',
                      dbDir, dbName, nside, 1, j, diffflux, seasnum,outDir)
    
    metric = SNMetric(config=config, x0_norm=x0_tab)
    for io, healpixID in enumerate(healpixels):
        obsMatch = obsFocalPlane.matchFast(healpixID)
        metric.run(season(obsMatch))

    if metric.save_status:
        metric.simu.Finish()
    """
        for key in resfi.keys():
            if resdict[key] is not None:
                if resfi[key] is None:
                    resfi[key] = resdict[key]
                else:
                    #print(key,resfi[key],resdict[key])
                    resfi[key] = np.concatenate((resfi[key],resdict[key]))

    if output_q is not None:
        return output_q.put({j:resfi})
    else:
        return resfi
    """


parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='',
                  help="output dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--diffflux", type="int", default=0,
                  help="flag for diff flux[%default]")
parser.add_option("--season", type="int", default=1,
                  help="season to process[%default]")

opts, args = parser.parse_args()

print('Start processing...')


# Load file

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
outDir = opts.outDir
if outDir == '':
    outDir = ' /sps/lsst/users/gris/Output_Simu_pipeline'


dbName = opts.dbName

nside = opts.nside
seasons = -1
nproc = opts.nproc
diffflux = opts.diffflux
seasnum = opts.season

# List of (instance of) metrics to process
metricList = []

# Load the (config) yaml file
config = makeYaml('input/param_simulation_gen.yaml',
                  dbDir, dbName, nside, 1, -1, diffflux,seasnum,outDir)

# check whether X0_norm file exist or not (and generate it if necessary)
absMag = config['SN parameters']['absmag']
salt2Dir = config['SN parameters']['salt2Dir']
model = config['Simulator']['model']
version = str(config['Simulator']['version'])

x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

if not os.path.isfile(x0normFile):
    from sn_tools.sn_utils import X0_norm
    X0_norm(salt2Dir=salt2Dir, model=model, version=version,
            absmag=absMag, outfile=x0normFile)

x0_tab = np.load(x0normFile)

# metricList.append(SNMetric(config=config,x0_norm=x0_tab))

# loading observations

observations = np.load('{}/{}.npy'.format(dbDir, dbName))
observations = renameFields(observations)

# this is a "simple" tessalation using healpix
pixels = pixelate(observations, nside, RaCol='fieldRA', DecCol='fieldDec')

# remove pixels with a "too-high" E(B-V)
idx = pixels['ebv']<= 0.25 
pixels = pixels[idx]

# this is a more complicated tessalation using a LSST Focal Plane
# Get the shape to identify overlapping obs
shape = GetShape(nside)
scanzone = shape.shape()
##
obsFocalPlane = ObsPixel(nside=nside, data=observations,
                         scanzone=scanzone, RaCol='fieldRA', DecCol='fieldDec')

#healpixels = np.unique(obsPixels['healpixID'])[:10]
#res = loop(healpixels, obsFocalPlane, band, metricList)

# print(res)

timeref = time.time()

healpixels = np.unique(pixels['healpixID'])
npixels = int(len(healpixels))
delta = npixels
if nproc > 1:
    delta = int(delta/(nproc))

tabpix = range(0, npixels, delta)
if npixels not in tabpix:
    tabpix = np.append(tabpix, npixels)

tabpix = tabpix.tolist()

if tabpix[-1]-tabpix[-2] <= 100:
    tabpix.remove(tabpix[-2])

print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()
for j in range(len(tabpix)-1):
    # for j in range(5,6):
    ida = tabpix[j]
    idb = tabpix[j+1]
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(
        healpixels[ida:idb], obsFocalPlane, dbDir, dbName, outDir, nside, diffflux,seasnum,x0_tab, j, result_queue))
    p.start()


"""
resultdict = {}
for i in range(len(tabpix)-1):
    resultdict.update(result_queue.get())

for p in multiprocessing.active_children():
    p.join()

restot = {}
for metric in metricList:
    restot[metric.name] = None

for key,vals in resultdict.items():
    for keyb in vals.keys():
        if restot[keyb] is None:
            restot[keyb] = vals[keyb]
        else:
            restot[keyb] = np.concatenate((restot[keyb],vals[keyb]))
"""
"""
for key, vals in restot.items():
    np.save('{}/{}_{}_{}.npy'.format(outDir,dbName,key,band),np.copy(vals))
"""
print('Done', time.time()-timeref)
