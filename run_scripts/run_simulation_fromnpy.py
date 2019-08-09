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
from sn_tools.sn_utils import GetReference
import pandas as pd

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
            print(np.unique(obs[['fieldId','note']]))
            idx = obs['fieldId'] == 0
            return pixelate(obs[idx], nside, RaCol='fieldRA', DecCol='fieldDec')
        else:
            """this is difficult
               we do not have other ways to identify
               DD except by selecting pixels with a large number of visits
            """
            pixels = pixelate(obs, nside, RaCol='fieldRA', DecCol='fieldDec')
            
            df = pd.DataFrame(np.copy(pixels))

            groups = df.groupby('healpixID').filter(lambda x: len(x)>5000)

            group_DD = groups.groupby(['fieldRA','fieldDec']).filter(lambda x: len(x)>4000)


            #return np.array(group_DD.to_records().view(type=np.matrix))
            return group_DD.to_records(index=False)


def makeYaml(input_file, dbDir, dbName, nside, nproc, num, diffflux,seasnum,outDir,fieldtype,x1,color,zmin,zmax,simu):

    with open(input_file, 'r') as file:
        filedata = file.read()

    prodid = '{}_{}_{}_seas_{}_{}_{}_{}'.format(simu,fieldtype,dbName, seasnum, num, x1,color)
    fullDbName = '{}/{}.npy'.format(dbDir, dbName)
    filedata = filedata.replace('prodid', prodid)
    filedata = filedata.replace('fullDbName', fullDbName)
    filedata = filedata.replace('nnproc', str(nproc))
    filedata = filedata.replace('nnside', str(nside))
    filedata = filedata.replace('outputDir', outDir)
    filedata = filedata.replace('diffflux', str(diffflux))
    filedata = filedata.replace('seasval', str(seasnum))
    filedata = filedata.replace('ftype', fieldtype)
    filedata = filedata.replace('x1val', str(x1))
    filedata = filedata.replace('colorval', str(color))
    filedata = filedata.replace('zmin', str(zmin))
    filedata = filedata.replace('zmax', str(zmax))


    if fieldtype == 'DD':
        filedata = filedata.replace('fcoadd', 'True')
    else:
        filedata = filedata.replace('fcoadd', 'False') 
    return yaml.load(filedata, Loader=yaml.FullLoader)


def loop(x1,color,zmin,zmax,healpixels, obsFocalPlane, dbDir, dbName, fieldtype, outDir, nside, diffflux,seasnum,x0_tab,reference_lc,simu,j=0, output_q=None):

    config = makeYaml('input/param_simulation_gen.yaml',
                      dbDir, dbName, nside, 8, j, diffflux, seasnum,outDir,fieldtype,x1,color,zmin,zmax,simu)
    
    metric = SNMetric(config=config, x0_norm=x0_tab,reference_lc=reference_lc)
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
parser.add_option("--fieldtype", type="str", default='DD', help="field - DD or WFD[%default]")
parser.add_option("--x1", type="float", default='0.0',
                  help="x1 SN [%default]")
parser.add_option("--color", type="float", default='0.0',
                  help="color SN [%default]")
parser.add_option("--zmin", type="float", default='0.0',
                  help="min redshift [%default]")
parser.add_option("--zmax", type="float", default='1.0',
                  help="max redshift [%default]")

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

nside = opts.nside
seasons = -1
nproc = opts.nproc
diffflux = opts.diffflux
seasnum = opts.season
fieldtype = opts.fieldtype
x1 = opts.x1
color = opts.color
zmin = opts.zmin
zmax = opts.zmax

simu = 'sncosmo'
# List of (instance of) metrics to process
metricList = []

# Load the (config) yaml file
config = makeYaml('input/param_simulation_gen.yaml',
                  dbDir, dbName, nside, 1, -1, diffflux, 
                  seasnum, outDir, fieldtype, x1, color,
                  zmin,zmax,simu)

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
    print('Loading reference LCs from',config['Simulator']['Reference File'])
    reference_lc = GetReference(
        config['Simulator']['Reference File'],
        config['Simulator']['Gamma File'],
        config['Instrument'])
    print('Reference LCs loaded')

# loading observations

observations = np.load('{}/{}.npy'.format(dbDir, dbName))
observations = renameFields(observations)


if fieldtype == 'DD':
    pixels = selectDD(observations,nside)
else:
    pixels = pixelate(observations, nside, RaCol='fieldRA', DecCol='fieldDec')

print('pixellisation done',type(pixels))
# get the number of visits per pixels
print(len(np.unique(pixels['healpixID'])))

# remove pixels with a "too-high" E(B-V)

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


#healpixels = np.unique(obsPixels['healpixID'])[:10]
#res = loop(healpixels, obsFocalPlane, band, metricList)

# print(res)

timeref = time.time()

healpixels = np.unique(pixels['healpixID'])
npixels = int(len(healpixels))
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


print(tabpix, len(tabpix))
result_queue = multiprocessing.Queue()
for j in range(len(tabpix)-1):
    # for j in range(5,6):
    ida = tabpix[j]
    idb = tabpix[j+1]
    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(x1,color,zmin,zmax,
        healpixels[ida:idb], obsFocalPlane, dbDir, dbName, fieldtype, outDir, nside, 
        diffflux,seasnum,x0_tab,reference_lc,simu, j, result_queue))
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
