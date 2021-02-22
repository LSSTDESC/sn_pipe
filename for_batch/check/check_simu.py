import glob
import pandas as pd
from optparse import OptionParser
from sn_tools.sn_io import loopStack
import numpy as np


def decode_name(thename):
    """
    Function to decode the name and extract infos

    Parameters
    ---------------
    thename: str
      the name to analyze

    Returns
    ----------
    RAmin, RAmax, Decmin, Decmax, npixels

    """

    fichname = thename.split('/')[-1]
    spl = fichname.split('_')
    RAmin = spl[11]
    RAmax = spl[12]
    Decmin = spl[13]
    Decmax = spl[14]

    npixels = int(spl[-2])

    return RAmin, RAmax, Decmin, Decmax, npixels


def ana_db(theDir, dbName):
    """
    Function to analyse the results of a dbName

    Parameters
    ---------------
    theDir: str
      location dir of the files
    dbName: str
      name of the db to analyze

    """

    fullName = '{}/{}/Simu*'.format(theDir, dbName)
    fis = glob.glob(fullName)

    dict_files = {}
    dict_npixels = {}
    for fi in fis:
        RAmin, RAmax, Decmin, Decmax, npixels = decode_name(fi)
        key = (RAmin, RAmax, Decmin, Decmax)
        if key not in dict_files.keys():
            dict_files[key] = []
        if key not in dict_npixels.keys():
            dict_npixels[key] = npixels
        dict_files[key].append(fi)

    for key, vals in dict_files.items():
        tab = loopStack(vals, 'astropyTable')
        print(key, len(np.unique(tab['healpixID']))/dict_npixels[key])


parser = OptionParser()

parser.add_option("--listFiles", type=str, default='WFD_simu.csv',
                  help="list of csv files containing list of db names [%default]")
parser.add_option("--theDir", type=str, default='../../Simulations_sncosmo',
                  help="location dir of dbs [%default]")

opts, args = parser.parse_args()

listFiles = opts.listFiles.split(',')

for ll in listFiles:
    print(ll)
    df = pd.read_csv(ll)
    print(df)
    for io, row in df.iterrows():
        ana_db(opts.theDir, row['dbName'])
