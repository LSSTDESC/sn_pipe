import pprint
import yaml
import argparse
import time
import h5py
from astropy.table import Table, vstack
import numpy as np
import multiprocessing
import queue
from optparse import OptionParser
from sn_fit.process_fit import Fitting
from sn_fit.mbcov import MbCov
import glob
import os
import sn_fit_input as simu_fit
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_io import loopStack, check_get_dir
from sn_fit_wrapper.sn_wrapper_for_fit import FitWrapper
from sn_tools.sn_io import checkDir


def load_lc_as_list(dirSimu, prodidSimu):
    """
    Function to make a list of lcs from astropy tables

    Parameters
    ----------
    dirSimu : str
        LC dir.
    prodidSimu : str
        LC tag.

    Returns
    -------
    res : list(LC)
        List of light curves.

    """

    search_path = '{}/LC_{}*.hdf5'.format(dirSimu, prodidSimu)
    print('searching', search_path)
    lc_files = glob.glob(search_path)

    res = []
    for lc_name in lc_files:
        fFile = h5py.File(lc_name, 'r')
        keys = fFile.keys()
        # print(keys)
        for key in keys:
            if 'table_column_meta' not in key:
                lc = Table.read(lc_name, path=key)
                lc.convert_bytestring_to_unicode()
                # print(lc)
                res.append(lc)

    return res


# get all possible simulation parameters and put in a dict
path = simu_fit.__path__
confDict = make_dict_from_config(path[0], 'config_fit.txt')

parser = argparse.ArgumentParser(
    description='Run a LC fitter on a set of LC curves.')

parser = OptionParser()
# parser for fit parameters : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

opts, args = parser.parse_args()

print('Start processing...')

# load the new values
newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key] = (vals[0], newval)

# new dict with configuration params
yaml_params = make_dict_from_optparse(newDict)


"""
covmb = None
mbCalc = yaml_params['mbcov']['estimate']

if mbCalc:
    # for this we need to have the SALT2 dir and files
    # if it does not exist get it from the web

    salt2Dir = yaml_params['mbcov']['directory']
    webPath = yaml_params['WebPathFit']
    check_get_dir(webPath, salt2Dir, salt2Dir)
    covmb = MbCov(salt2Dir, paramNames=dict(
        zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))
"""
# create outputdir if necessary
outDir = yaml_params['OutputFit']['directory']
checkDir(outDir)

prodid = yaml_params['ProductionIDFit']
yaml_name = '{}/Fit_{}.yaml'.format(outDir, prodid)
with open(yaml_name, 'w') as f:
    data = yaml.dump(yaml_params, f)
# get the simu files
dirSimu = yaml_params['Simulations']['dirname']
prodidSimu = yaml_params['Simulations']['prodid']

print('dirsimu', dirSimu, prodidSimu)
list_lc = load_lc_as_list(dirSimu, prodidSimu)

time_ref = time.time()
fit_wrapper = FitWrapper(yaml_params)
fitlc = fit_wrapper(list_lc)
fit_wrapper.dump(fitlc)
ccols = ['SNID', 'x1', 'color', 'x1_fit', 'color_fit']
print('after all fits', time.time()-time_ref)
pprint.pprint(fitlc[ccols])
# now fit all this
# Fit_Simu(yaml_params, covmb)
