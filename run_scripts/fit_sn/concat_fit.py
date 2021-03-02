from optparse import OptionParser
import glob
from astropy.table import Table, vstack
import h5py
import pandas as pd
import multiprocessing
import numpy as np
import os

def loadFile(filename):
    """
    Function to load a file according to the type of data it contains

    Parameters
    ---------------
    filename: str
       name of the file to consider

    Returns
    -----------
    object of the file (stacked)

    """

    # open the file
    f = h5py.File(filename, 'r')
    # get the keys
    keys = f.keys()
    
    # put the result in res
    res = Table()

    for kk in keys:
        # loop on the keys and concat objects
        tab  = Table.read(filename, path=kk)
        if len(tab)>0:
            res = vstack([res, tab], metadata_conflicts='silent')
            

    return res


def getpatch(fi,dbName,dict_patches):
    """
    Function to estimate (Ra,Dec) ,coords of the path the file is belonging to

    Parameters
    ----------
    fi: str
      fileName to analyze
    dbName: str
      OS name of interest
    dict_patches: dict
      dict of patches and corresponding files

    Returns
    -------
    updated dict_patches

    """

    finame = fi.split('/')[-1]
    fisplit = finame.split(dbName)[-1].split('_')
    prefix = finame.split(dbName)[0]

    RAmin = fisplit[5]
    RAmax = fisplit[6]
    
    Decmin = fisplit[7]
    Decmax = fisplit[8]
    key = (RAmin,RAmax,Decmin,Decmax)
    key = (RAmin,RAmax)
    if key not in dict_patches.keys():
        dict_patches[key] = []
    dict_patches[key].append(fi)


    return dict_patches,prefix

def concat(prefix,dbName,key, files,outDir,nproc=8):
    """
    Function to concatenate astropytables

    Parameters
    ----------
    prefix: str,
      prefix for outputName
    dbName: str
      OS of interest
    key: str
      key used in the outputname
    files: list(str)
     list of files to stack
    nproc: int
     number of procs for multiprocessing

    """
           
    # multiprocessing parameters
    nz = len(files)
    t = np.linspace(0, nz, nproc+1, dtype='int')
       
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=concat_list,
                                         args=(files[t[j]:t[j+1]], j, result_queue))
                 for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict

    for i in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = pd.DataFrame()

    # gather the results
    for keyb, vals in resultdict.items():
        restot = pd.concat((restot, vals), sort=False)

    #write this output file
    restot = restot[restot.columns.drop(list(restot.filter(regex='mask')))]
    print(restot.columns,key,dbName,prefix)
    outName = '{}/{}{}_{}.hdf5'.format(outDir,prefix,dbName,'_'.join(kk for kk in key))
    print(outName)

    print(restot.dtypes)
    Table.from_pandas(restot).write(outName,'summary')
    

def concat_list(files, j=0, output_q=None):
    
    restab = pd.DataFrame()
    for fi in files:
        tab = loadFile(fi)
        #restab = vstack([restab,tab])
        restab = pd.concat((restab,tab.to_pandas()))

    if output_q is not None:
        return output_q.put({j: restab})
    else:
        return restab


parser = OptionParser(
    description='script to concatenate files from fitted sn')
parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/Fit_sncosmo/',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='baseline_v1.5_10yrs',
                  help="OS name to process [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/web/PV_ABRH/files',
                  help="Ouput dir [%default]")

opts, args = parser.parse_args()

dirFile = opts.dirFile
dbName = opts.dbName
outDir = opts.outDir

#create outDir if does not exist
if not os.path.isdir(outDir):
    os.makedirs(outDir)


fullName= '{}/{}/Fit*.hdf5'.format(dirFile,dbName)

fis = glob.glob(fullName)
print(fis)

dict_patches = {}
prefix =''
for fi in fis:
    dict_patches,prefix = getpatch(fi,dbName,dict_patches)

print(len(dict_patches.keys()))

for key, vals in dict_patches.items():
    print(key,len(vals))
    #if key[0]=='144.0':
    concat(prefix,dbName,key,vals,outDir)
    #break
