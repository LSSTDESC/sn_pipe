from simuWrapper import SimuWrapper, MakeYaml
import sn_simu_input as simu_input
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse
from sn_tools.sn_process import Process
from optparse import OptionParser
import numpy as np
import os
import yaml


# get all possible simulation parameters and put in a dict
path = simu_input.__path__
confDict = make_dict_from_config(path[0],'config_simulation.txt')


parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str",
                  default=' /sps/lsst/cadence/LSST_SN_CADENCE/cadence_db', help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--nodither", type="int", default=0,
                  help="to remove dithering [%default]")
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
parser.add_option("--npixels", type=int, default=-1,
                  help="number of pixels to process[%default]")
parser.add_option("--nclusters", type=int, default=0,
                  help="number of clusters in data (DD only)[%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius around clusters (DD and Fakes)[%default]")
parser.add_option("--nproc", type=int, default=1,
                  help="number of procs to run[%default]")

# parser for simulation parameters : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')

opts, args = parser.parse_args()

print('Start processing...')

#load the new values
newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)

# new dict with configuration params
yaml_params = make_dict_from_optparse(newDict)

# one modif: full dbName
yaml_params['Observations']['filename']='{}/{}.{}'.format(opts.dbDir, opts.dbName, opts.dbExtens)

"""
# Generate the yaml file
# build the yaml file

makeYaml = MakeYaml(opts.dbDir, opts.dbName, opts.dbExtens, opts.nside,
                    opts.nprocsimu, opts.diffflux, opts.season, opts.outDir,
                    opts.fieldType,
                    opts.x1Type, opts.x1min, opts.x1max, opts.x1step,
                    opts.colorType, opts.colormin, opts.colormax, opts.colorstep,
                    opts.zType, opts.zmin, opts.zmax, opts.zstep,
                    opts.simulator, opts.daymaxType, opts.daymaxstep,
                    opts.coadd, opts.prodid, opts.ebvofMW, opts.bluecutoff, opts.redcutoff,opts.error_model)

yaml_orig = 'input/simulation/param_simulation_gen.yaml'

yaml_params = makeYaml.genYaml(yaml_orig)
"""
print(yaml_params)

# save on disk

# create outputdir if does not exist
outDir = yaml_params['Output']['directory']
prodid = yaml_params['ProductionID']

if not os.path.isdir(outDir):
    os.makedirs(outDir)

yaml_name = '{}/{}.yaml'.format(outDir, prodid)
with open(yaml_name, 'w') as f:
    data = yaml.dump(yaml_params, f)

# define what to process using simuWrapper

metricList = [SimuWrapper(yaml_name)]
fieldType = yaml_params['Observations']['fieldtype']
fieldName = yaml_params['Observations']['fieldname']
nside = yaml_params['Pixelisation']['nside']
saveData = 0
outDir = yaml_params['Output']['directory']
# now perform the processing

Process(opts.dbDir, opts.dbName, opts.dbExtens,
        fieldType, fieldName,nside,
        opts.RAmin, opts.RAmax,
        opts.Decmin, opts.Decmax,
        saveData, opts.remove_dithering,
        outDir, opts.nproc, metricList,
        opts.pixelmap_dir, opts.npixels,
        opts.nclusters, opts.radius)
