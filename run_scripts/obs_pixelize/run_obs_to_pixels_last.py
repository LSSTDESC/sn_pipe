from sn_tools.sn_process import FP2pixels
from optparse import OptionParser
import os

def outName(outDir='',dbName='',fieldType='',nside=128,RAmin=0., RAmax=360.,Decmin=-85,Decmax=40.,fieldName='',**kwargs):
    
    if fieldType == 'DD':
        outName_ = '{}/{}_{}_nside_{}_{}_{}_{}_{}_{}.hdf5'.format(outDir,
                                                                 dbName, fieldType, nside,
                                                                 RAmin, RAmax,
                                                                 Decmin, Decmax,
                                                                 fieldName)
    if fieldType == 'WFD':
        outName_ = '{}/{}_{}_nside_{}_{}_{}_{}_{}.hdf5'.format(outDir,
                                                             dbName, fieldType, nside,
                                                             RAmin, RAmax,
                                                             Decmin, Decmax)

    return outName_

def genDir(outDir,dbName,nodither):
    """
    Method to create directory
    """
    
    # prepare outputDir
    nodither = ''
    if nodither==1:
        nodither = '_nodither'

    outDir = '{}/{}{}'.format(outDir,dbName, nodither)

    # create output directory (if necessary)

    if not os.path.isdir(outDir):
        os.makedirs(outDir)

    return outDir
    
        
parser = OptionParser()

parser.add_option("--dbName", type="str", default='draft_connected_v2.99_10yrs',
                  help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str",
                  default='../DB_Files', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='ObsPixelized',
                  help="output dir [%default]")
parser.add_option("--nside", type="int", default=128,
                  help="healpix nside [%default]")
parser.add_option("--nproc", type="int", default=1,
                  help="number of procs  [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type DD or WFD[%default]")
parser.add_option("--remove_dithering", type="int", default=0,
                  help="remove dithering for DDF [%default]")
parser.add_option("--saveData", type="int", default=1,
                  help="flag to dump data on disk [%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - [%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - [%default]")
parser.add_option("--Decmin", type=float, default=-90.,
                  help="Dec min for obs area - [%default]")
parser.add_option("--Decmax", type=float, default=40.,
                  help="Dec max for obs area - [%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="verbose mode for the metric[%default]")
parser.add_option("--fieldName", type='str', default='COSMOS',
                  help="fieldName [%default]")
parser.add_option("--radius", type=float, default=4.,
                  help="radius around center - for DD only[%default]")
parser.add_option("--VRO_FP", type=str, default='circular',
                  help="VRO Focal Plane (circle or realistic) [%default]")
parser.add_option("--project_FP", type=str, default='gnomonic',
                  help="Focal Plane projection (gnomonic or hp_query) [%default]")
parser.add_option("--telrot", type=int, default=0,
                  help="telescope rotation angle [%default]")
parser.add_option("--display", type=int, default=0,
                  help="to display results [%default]")
parser.add_option("--npixels", type="int", default=-1,
                  help="number of pixels [%default]")

opts, args = parser.parse_args()


fieldNames = opts.fieldName.split(',')
dict_process = vars(opts)


if opts.saveData:
    outDir = genDir(opts.outDir,opts.dbName,opts.remove_dithering)

for fieldName in fieldNames:
    dict_process['fieldName']=fieldName
    obs2pixels = FP2pixels(**dict_process)

    pixels = obs2pixels()

    print(type(pixels),outDir)
    if opts.saveData:
        dict_process['outDir'] = outDir
        outName_= outName(**dict_process)
        print('outName',outName_)
        pixels.to_hdf(outName_,key=fieldName)
