from simuWrapper import SimuWrapper, MakeYaml
from optparse import OptionParser
from sn_tools.sn_process import Process
import numpy as np
import os
import yaml

parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str",
                  default=' /sps/lsst/cadence/LSST_SN_CADENCE/cadence_db', help="db dir [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--outDir", type="str", default='Output_Simu',
                  help="output dir [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of proc  [%default]")
parser.add_option("--diffflux", type=int, default=0,
                  help="flag for diff flux[%default]")
parser.add_option("--season", type="int", default=-1,
                  help="season to process[%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field - DD or WFD[%default]")
parser.add_option("--x1Type", type="str", default='unique',
                  help="x1 type (unique,random,uniform) [%default]")
parser.add_option("--x1min", type=float, default=-2.0,
                  help="x1 min if x1Type=unique (x1val) or uniform[%default]")
parser.add_option("--x1max", type=float, default=2.0,
                  help="x1 max - if x1Type=uniform[%default]")
parser.add_option("--x1step", type=float, default=0.1,
                  help="x1 step - if x1Type=uniform[%default]")
parser.add_option("--colorType", type="str", default='unique',
                  help="color type (unique,random,uniform) [%default]")
parser.add_option("--colormin", type=float, default=0.2,
                  help="color min if colorType=unique (colorval) or uniform[%default]")
parser.add_option("--colormax", type=float, default=0.3,
                  help="color max - if colorType=uniform[%default]")
parser.add_option("--colorstep", type=float, default=0.1,
                  help="color step - if colorType=uniform[%default]")
parser.add_option("--zType", type="str", default='uniform',
                  help=" zcolor type (unique,uniform,random) [%default]")
parser.add_option("--daymaxType", type="str", default='unique',
                  help="daymax type (unique,uniform,random) [%default]")
parser.add_option("--daymaxstep", type=float, default=1,
                  help="daymax step [%default]")
parser.add_option("--zmin", type="float", default=0.0,
                  help="min redshift [%default]")
parser.add_option("--zmax", type="float", default=1.0,
                  help="max redshift [%default]")
parser.add_option("--zstep", type="float", default=0.02,
                  help="max redshift [%default]")
parser.add_option("--saveData", type="int", default=0,
                  help="to save data [%default]")
parser.add_option("--nodither", type="int", default=0,
                  help="to remove dithering [%default]")
parser.add_option("--coadd", type="int", default=1,
                  help="to coadd or not[%default]")
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
parser.add_option("--simulator", type=str, default='sn_cosmo',
                  help="simulator to use[%default]")
parser.add_option("--prodid", type=str, default='Test',
                  help="prod id tag[%default]")
parser.add_option("--ebvofMW", type=float, default=-1.,
                  help="ebvofMW value[%default]")

opts, args = parser.parse_args()

print('Start processing...')

# Generate the yaml file
# build the yaml file

makeYaml = MakeYaml(opts.dbDir, opts.dbName, opts.dbExtens, opts.nside,
                    opts.nproc, opts.diffflux, opts.season, opts.outDir,
                    opts.fieldType,
                    opts.x1Type, opts.x1min, opts.x1max, opts.x1step,
                    opts.colorType, opts.colormin, opts.colormax, opts.colorstep,
                    opts.zType, opts.zmin, opts.zmax, opts.zstep,
                    opts.simulator, opts.daymaxType, opts.daymaxstep,
                    opts.coadd, opts.prodid, opts.ebvofMW)

yaml_orig = 'input/simulation/param_simulation_gen.yaml'

yaml_params = makeYaml.genYaml(yaml_orig)

print(yaml_params)

# save on disk

# create outputdir if does not exist
if not os.path.isdir(opts.outDir):
    os.makedirs(opts.outDir)

yaml_name = '{}/{}.yaml'.format(opts.outDir, opts.prodid)
with open(yaml_name, 'w') as f:
    data = yaml.dump(yaml_params, f)

# define what to process using simuWrapper

metricList = [SimuWrapper(yaml_name)]

# now perform the processing

Process(opts.dbDir, opts.dbName, opts.dbExtens,
        opts.fieldType, opts.nside,
        opts.RAmin, opts.RAmax,
        opts.Decmin, opts.Decmax,
        opts.saveData, opts.remove_dithering,
        opts.outDir, opts.nproc, metricList,
        opts.pixelmap_dir, opts.npixels,
        opts.nclusters, opts.radius)
