from simuWrapper import SimuWrapper, MakeYaml
from optparse import OptionParser
from sn_tools.sn_process import Process
import numpy as np
import os
import yaml

parser = OptionParser()

parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - for WDF only[%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - for WDF only[%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - for WDF only[%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - for WDF only[%default]")
parser.add_option("--remove_dithering", type="int", default='0',
                  help="remove dithering for DDF [%default]")
parser.add_option("--pixelmap_dir", type=str, default='',
                  help="dir where to find pixel maps[%default]")
parser.add_option("--npixels", type=int, default=0,
                  help="number of pixels to process[%default]")
parser.add_option("--nclusters", type=int, default=0,
                  help="number of clusters in data (DD only)[%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius around clusters (DD and Fakes)[%default]")
parser.add_option("--config_yaml", type=str, default='',
                  help="input yaml config file[%default]")

opts, args = parser.parse_args()

print('Start processing...')

# define what to process using simuWrapper

metricList = [SimuWrapper(opts.config_yaml)]

# extract yaml infos for Process
with open(opts.config_yaml) as file:
    config = yaml.full_load(file)

dbDir = '/'.join(config['Observations']['filename'].split('/')[:-1])
if dbDir == '':
    dbDir = '.'

dbName = config['Observations']['filename'].split('/')[-1]
dbExtens = dbName.split('.')[-1]
dbName = '.'.join(dbName.split('.')[:-1])
fieldType = config['Observations']['fieldtype']
nside = config['Pixelisation']['nside']
outDir = config['Output']['directory']
nproc = config['Multiprocessing']['nproc']
print(dbDir, dbName, dbExtens, fieldType, nside)
prodid = config['ProductionID']


# save on disk

# create outputdir if does not exist
if not os.path.isdir(outDir):
    os.makedirs(outDir)

yaml_name = '{}/{}.yaml'.format(outDir, prodid)
with open(yaml_name, 'w') as f:
    data = yaml.dump(config, f)


# now perform the processing
Process(dbDir, dbName, dbExtens,
        fieldType, nside,
        opts.RAmin, opts.RAmax,
        opts.Decmin, opts.Decmax,
        False, opts.remove_dithering,
        outDir, nproc, metricList,
        opts.pixelmap_dir, opts.npixels,
        opts.nclusters, opts.radius)
