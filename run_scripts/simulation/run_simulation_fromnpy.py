import matplotlib
matplotlib.use('agg')
import numpy as np
import healpy as hp
#from metricWrapper import CadenceMetricWrapper, SNRMetricWrapper
from sn_mafsim.sn_maf_simulation import SNMetric
#from sn_tools.sn_obs import renameFields, pixelate, season, GetShape, ObsPixel,ProcessArea
from sn_tools.sn_obs import renameFields, pixelate, season, ProcessArea
from optparse import OptionParser
import time
import multiprocessing
import yaml
import os
from sn_tools.sn_utils import GetReference
import pandas as pd
from sn_tools.sn_obs import PavingSky, getFields
import glob
from sn_tools.sn_io import getObservations
from sn_tools.sn_cadence_tools import ClusterObs
import numpy.lib.recfunctions as rf


def selectDD(obs, nside):

    print(obs.dtype)

    print(np.unique(obs['proposalId']))

    propId = list(np.unique(obs['proposalId']))

    if len(propId) > 3:
        idx = obs['proposalId'] == 5
        return pixelate(obs[idx], nside, RaCol='fieldRA', DecCol='fieldDec')
    else:
        names = obs.dtype.names
        if 'fieldId' in names:
            print(np.unique(obs[['fieldId', 'note']]))
            idx = obs['fieldId'] == 0
            return pixelate(obs[idx], nside, RaCol='fieldRA', DecCol='fieldDec')
        else:
            """this is difficult
               we do not have other ways to identify
               DD except by selecting pixels with a large number of visits
            """
            pixels = pixelate(obs, nside, RaCol='fieldRA', DecCol='fieldDec')

            df = pd.DataFrame(np.copy(pixels))

            groups = df.groupby('healpixID').filter(lambda x: len(x) > 5000)

            group_DD = groups.groupby(['fieldRA', 'fieldDec']).filter(
                lambda x: len(x) > 4000)

            # return np.array(group_DD.to_records().view(type=np.matrix))
            return group_DD.to_records(index=False)


def makeYaml(input_file, dbDir, dbName, nside, nproc,
             diffflux, seasnum, outDir, fieldType,
             x1, color, zmin, zmax, simu,
             x1colorType, zType, daymaxType):

    with open(input_file, 'r') as file:
        filedata = file.read()

    prodid = '{}_{}_{}_seas_{}_{}_{}'.format(
        simu, fieldType, dbName, seasnum, x1, color)
    fullDbName = '{}/{}.npy'.format(dbDir, dbName)
    filedata = filedata.replace('prodid', prodid)
    filedata = filedata.replace('fullDbName', fullDbName)
    filedata = filedata.replace('nnproc', str(nproc))
    filedata = filedata.replace('nnside', str(nside))
    filedata = filedata.replace('outputDir', outDir)
    filedata = filedata.replace('diffflux', str(diffflux))
    filedata = filedata.replace('seasval', str(seasnum))
    filedata = filedata.replace('ftype', fieldType)
    filedata = filedata.replace('x1val', str(x1))
    filedata = filedata.replace('colorval', str(color))
    filedata = filedata.replace('zmin', str(zmin))
    filedata = filedata.replace('zmax', str(zmax))
    filedata = filedata.replace('x1colorType', x1colorType)
    filedata = filedata.replace('zType', zType)
    filedata = filedata.replace('daymaxType', daymaxType)

    if fieldType == 'DD':
        filedata = filedata.replace('fcoadd', 'True')
    else:
        filedata = filedata.replace('fcoadd', 'False')
    return yaml.load(filedata, Loader=yaml.FullLoader)


# def loop_area(pointings,metricList, observations, nside, outDir,dbName,saveData,nodither,RaCol,DecCol,j=0, output_q=None):
def loop_area(pointings, config, x0_tab, reference_lc, coadd, observations, nside, outDir, dbName, saveData, nodither, RaCol, DecCol, j=0, output_q=None):

    # this is to take multiproc into account
    config['ProductionID'] = '{}_{}'.format(config['ProductionID'], j)
    metric = SNMetric(config=config, x0_norm=x0_tab,
                      reference_lc=reference_lc, coadd=coadd)

    print('processing pointings', j, pointings, observations.dtype)

    time_ref = time.time()

    ipoint = 1

    myprocess = ProcessArea(nside, RaCol, DecCol, j, outDir, dbName, saveData)
    for pointing in pointings:
        ipoint += 1

        resdict = myprocess(observations, [metric], pointing['Ra'], pointing['Dec'],
                            pointing['radius'], pointing['radius'], ipoint, nodither, display=False)

    if hasattr(metric, 'type'):
        if metric.type == 'simulation':
            metric.simu.Finish()
    print('end of processing for', j, time.time()-time_ref)


def loop(x1, color, zmin, zmax, healpixels, obsFocalPlane, dbDir, dbName, fieldType, outDir, nside, diffflux, seasnum, x0_tab, reference_lc, simu, j=0, output_q=None):

    config = makeYaml('input/simulation/param_simulation_gen.yaml',
                      dbDir, dbName, nside, 8, diffflux, seasnum, outDir, fieldType, x1, color, zmin, zmax, simu)

    metric = SNMetric(config=config, x0_norm=x0_tab, reference_lc=reference_lc)
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
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--outDir", type="str", default='',
                  help="output dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type="int", default='8',
                  help="number of proc  [%default]")
parser.add_option("--diffflux", type="int", default=0,
                  help="flag for diff flux[%default]")
parser.add_option("--season", type="int", default=-1,
                  help="season to process[%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field - DD or WFD[%default]")
parser.add_option("--x1colorType", type="str", default='fixed',
                  help="x1color type (fixed,random) [%default]")
parser.add_option("--zType", type="str", default='uniform',
                  help="x1color type (fixed,uniform,random) [%default]")
parser.add_option("--daymaxType", type="str", default='unique',
                  help="x1color type (unique,uniform,random) [%default]")
parser.add_option("--x1", type="float", default='0.0',
                  help="x1 SN [%default]")
parser.add_option("--color", type="float", default='0.0',
                  help="color SN [%default]")
parser.add_option("--zmin", type="float", default='0.0',
                  help="min redshift [%default]")
parser.add_option("--zmax", type="float", default='1.0',
                  help="max redshift [%default]")
parser.add_option("--saveData", type="int", default=1,
                  help="to save data [%default]")
parser.add_option("--nodither", type="int", default=0,
                  help="to remove dithering [%default]")
parser.add_option("--coadd", type="int", default=1,
                  help="to coadd or not[%default]")
parser.add_option("--ramin", type=float, default=0.,
                  help="ra min for obs area - for WDF only[%default]")
parser.add_option("--ramax", type=float, default=360.,
                  help="ra max for obs area - for WDF only[%default]")
parser.add_option("--decmin", type=float, default=-1.,
                  help="dec min for obs area - for WDF only[%default]")
parser.add_option("--decmax", type=float, default=-1.,
                  help="dec max for obs area - for WDF only[%default]")

opts, args = parser.parse_args()

print('Start processing...')


# Load file

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'
outDir = opts.outDir
if outDir == '':
    outDir = '/sps/lsst/users/gris/Output_Simu_pipeline'


dbName = opts.dbName
dbExtens = opts.dbExtens
nside = opts.nside
seasons = -1
nproc = opts.nproc
diffflux = opts.diffflux
seasnum = opts.season
fieldType = opts.fieldType
x1colorType = opts.x1colorType
zType = opts.zType
daymaxType = opts.daymaxType
x1 = opts.x1
color = opts.color
zmin = opts.zmin
zmax = opts.zmax
saveData = opts.saveData
nodither = opts.nodither
coadd = opts.coadd
ramin = opts.ramin
ramax = opts.ramax
decmin = opts.decmin
decmax = opts.decmax

simu = 'sncosmo'
# List of (instance of) metrics to process
#metricList = []

# Load the (config) yaml file
config = makeYaml('input/simulation/param_simulation_gen.yaml',
                  dbDir, dbName, nside, 1, diffflux,
                  seasnum, outDir, fieldType, x1, color,
                  zmin, zmax, simu, x1colorType, zType, daymaxType)

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

reference_lc = None
if 'sn_fast' in config['Simulator']['name']:
    simu = 'sn_fast'
    print('Loading reference LCs from', config['Simulator']['Reference File'])
    reference_lc = GetReference(
        config['Simulator']['Reference File'],
        config['Simulator']['Gamma File'],
        config['Instrument'])
    print('Reference LCs loaded')

# loading observations

observations = getObservations(dbDir, dbName, dbExtens)

# rename fields

observations = renameFields(observations)

dictArea = {}
radius = 5.
RaCol = 'fieldRA'
DecCol = 'fieldDec'
if 'Ra' in observations.dtype.names:
    RaCol = 'Ra'
    DecCol = 'Dec'

if fieldType == 'DD':
    n_clusters = 5
    if 'euclid' in dbName:
        n_clusters = 6

    fieldIds = [290, 744, 1427, 2412, 2786]
    observations = getFields(observations, fieldType, fieldIds, nside)

    # get clusters out of these obs

    clusters = ClusterObs(
        observations, n_clusters=n_clusters, dbName=dbName).clusters

    clusters = rf.append_fields(clusters, 'radius', [radius]*len(clusters))

    areas = rf.rename_fields(clusters, {'RA': 'Ra'})

    print('yes', areas)
    print(observations.dtype)

else:
    if fieldType == 'WFD':
        observations = getFields(observations, 'WFD')
        minDec = decmin
        maxDec = decmax
        if decmin == -1.0 and decmax == -1.0:
            # in that case min and max dec are given by obs strategy
            minDec = np.min(observations['fieldDec'])-3.
            maxDec = np.max(observations['fieldDec'])+3.
        areas = PavingSky(ramin, ramax, minDec, maxDec, radius)
        #areas = pavingSky(20., 40., -40., -30., radius)
        print(observations.dtype)

    if fieldType == 'Fake':
        # in that case: only one (Ra,Dec)
        radius = 0.1
        Ra = np.unique(observations[RaCol])[0]
        Dec = np.unique(observations[DecCol])[0]
        areas = PavingSky(Ra-radius/2., Ra+radius/2., Dec -
                          radius/2., Dec+radius/2., radius)

# print(observations.dtype)
# print(test)


# remove pixels with a "too-high" E(B-V)
"""
if fieldtype != 'DD':
    idx = pixels['ebv']<= 0.25 
    pixels = pixels[idx]

# this is a more complicated tessalation using a LSST Focal Plane
# Get the shape to identify overlapping obs
shape = GetShape(nside)
scanzone = shape.shape()
##
obsFocalPlane = ObsPixel(nside=nside, data=observations,
                         scanzone=scanzone, RaCol='fieldRA', DecCol='fieldDec')
"""
metricList = []

config = makeYaml('input/simulation/param_simulation_gen.yaml',
                  dbDir, dbName, nside, 8, diffflux, seasnum,
                  outDir, fieldType, x1, color, zmin, zmax, simu,
                  x1colorType, zType, daymaxType)

with open('prod_yaml/result.yml', 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

#metricList.append(SNMetric(config=config, x0_norm=x0_tab,reference_lc=reference_lc,coadd=coadd))


#healpixels = np.unique(obsPixels['healpixID'])[:10]
#res = loop(healpixels, obsFocalPlane, band, metricList)

# print(res)

timeref = time.time()

healpixels = areas
npixels = int(len(healpixels))
"""
print('number of pixels',npixels)
delta = npixels
if nproc > 1:
    delta = int(delta/(nproc))

tabpix = range(0, npixels, delta)
if npixels not in tabpix:
    tabpix = np.append(tabpix, npixels)

tabpix = tabpix.tolist()
if nproc > 1:
    if tabpix[-1]-tabpix[-2] <= 100:
        tabpix.remove(tabpix[-2])
"""
tabpix = np.linspace(0, npixels, nproc+1, dtype='int')
print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()
for j in range(len(tabpix)-1):
    # for j in range(0,1):
    ida = tabpix[j]
    idb = tabpix[j+1]
    """
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(x1,color,zmin,zmax,
        healpixels[ida:idb], obsFocalPlane, dbDir, dbName, fieldtype, outDir, nside, 
        diffflux,seasnum,x0_tab,reference_lc,simu, j, result_queue)) 
   
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop_area, args=(
        healpixels[ida:idb],metricList, observations, nside,outDir,dbName, saveData,nodither,RaCol,DecCol,j, result_queue))
    """
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop_area, args=(
        healpixels[ida:idb], config, x0_tab, reference_lc, coadd, observations, nside, outDir, dbName, saveData, nodither, RaCol, DecCol, j, result_queue))

    p.start()
