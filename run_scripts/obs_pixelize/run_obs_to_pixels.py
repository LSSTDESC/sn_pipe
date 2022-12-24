from sn_tools.sn_process import FP2pixels
from optparse import OptionParser
import os
from sn_tools.sn_io import make_dict_from_config,add_parser

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
    
confDict_gen = make_dict_from_config('input/obs_pixelize', 'config_obs_pixelize.txt')

parser = OptionParser()
# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict_gen)

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
