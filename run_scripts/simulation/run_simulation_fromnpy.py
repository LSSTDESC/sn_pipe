import matplotlib
matplotlib.use('agg')
import numpy as np
from sn_tools.sn_process import Process
from optparse import OptionParser
from simuWrapper import SimuWrapper

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
parser.add_option("--simulator", type=str, default='sncosmo',
                  help="simulator to use[%default]")

opts, args = parser.parse_args()

print('Start processing...')

# define what to process using simuWrapper

metricList = [SimuWrapper(opts.dbDir, opts.dbName, opts.nside,
                          opts.nproc, opts.diffflux,
                          opts.season, opts.outDir,
                          opts.fieldType, opts.x1, opts.color,
                          opts.zmin, opts.zmax, opts.simulator,
                          opts.x1colorType, opts.zType,
                          opts.daymaxType)]

# now perform the processing
Process(opts.dbDir, opts.dbName, opts.dbExtens,
        opts.fieldType, opts.nside,
        opts.RAmin, opts.RAmax,
        opts.Decmin, opts.Decmax,
        opts.saveData, opts.remove_dithering,
        opts.outDir, opts.nproc, metricList,
        opts.pixelmap_dir, opts.npixels,
        opts.nclusters, opts.radius)
