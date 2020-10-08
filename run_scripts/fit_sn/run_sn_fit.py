import yaml
import argparse
import time
import h5py
from astropy.table import Table, vstack
import numpy as np
import multiprocessing
from optparse import OptionParser
from sn_fit.process_fit import Fitting
from sn_fit.mbcov import MbCov
import glob
import os
import sn_fit_input as simu_fit
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse
from sn_tools.sn_io import loopStack,check_get_dir

class Fit_Simu:
    """
    class to fit simulated LC
    
    Parameters
    --------------
    config: dict
      configuration parameters
    covmb: ??

    """
    def __init__(self,config, covmb):

        # Fit instance

        self.fit = Fitting(config, covmb=covmb)

        # get the simu files
        dirSimu = config['Simulations']['dirname']
        prodidSimu = config['Simulations']['prodid']
    
        search_path = '{}/Simu_{}*.hdf5'.format(dirSimu, prodidSimu)
        self.simu_files = glob.glob(search_path)

        # nproc (for multiprocessing)
        self.nproc=config['Multiprocessing']['nproc']
        # process here
        self.fitSimu()

    def fitSimu(self):
        """
        Method to perform the fit - loops on input files

        Parameters
        --------------
        j: int,opt
          tag for multiprocessing use (default: 0)
        output_queue: multiprocessing queue
          (default: None)

        Returns
        ----------
        
        
        """
        res = Table()
        idump = -1
        
        for name in self.simu_files:
            time_ref = time.time()
            namespl = name.split('/')
            last_name = namespl[-1]
            prefix = '/'.join([nn for nn in namespl[:-1]])
            lc_name = '{}/{}'.format(prefix,last_name.replace('Simu_','LC_'))
            f = h5py.File(name, 'r')
            # reading the simu file
            simul = Table()
            for i, key in enumerate(f.keys()):
                simul = vstack([simul, Table.read(name, path=key)])

            print('Number of LC to fit',len(simul))
            #the idea here is to split this file before multiprocessing
           

            nn = 1
            
            
            n_per_batch = int(len(simul)/self.nproc)
            if n_per_batch >= 300:
                nn = int(n_per_batch/300)
            
            t = np.linspace(0, len(simul), nn+1, dtype='int')

            print('hello here',t)
            
            for kk in range(nn):
            #for kk in [1]:
                simul_h = simul[t[kk]:t[kk+1]]
                res_simul = self.process_multiproc(simul_h,lc_name)
                res = vstack([res, res_simul])
                ## dumping in file
                print('dumping results', len(res))
                if len(res) > 0:
                    idump += 1
                    self.fit.dump(res, idump)
                    res = Table()

            if len(res) > 0:
                idump += 1
                self.fit.dump(res, idump)
                res = Table()
            print('done here', time.time()-time_ref)
       
    def process_multiproc(self,simul,lc_name):
        
        
        print('Number of LC to fit',len(simul))
        result_queue = multiprocessing.Queue()
        nlc = len(simul)
        t = np.linspace(0, nlc, self.nproc+1, dtype='int')
        print(t)

        
        procs = [multiprocessing.Process(name='Subprocess-'+str(i), target=self.process, args=(
            simul[t[i]:t[i+1]], lc_name,i, result_queue)) for i in range(self.nproc)]
        """
        procs = [multiprocessing.Process(name='Subprocess-'+str(i), target=self.process, args=(
            simul[t[i]:t[i+1]], lc_name,i, result_queue)) for i in [6]]
        """
        for p in procs:
            p.start()

        resultdict = {}
        for ja in range(self.nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        res = Table()
        for ja in range(self.nproc):
            res = vstack([res, resultdict[ja]])
            
        return res

    def process(self,simul, lc_name,j=0, output_q=None):

        res = Table()
        for ilc,simu in enumerate(simul):
            #print('fitting',j,ilc)
            lc = None
            lc = Table.read(lc_name, path='lc_{}'.format(simu['index_hdf5']))
            lc.convert_bytestring_to_unicode()
            if simu['status'] == 1:
                resfit = self.fit(lc)
                res = vstack([res, resfit])

        #print('done here',j)
        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res
        

# get all possible simulation parameters and put in a dict
path = simu_fit.__path__
confDict = make_dict_from_config(path[0],'config_simulation.txt')

parser = argparse.ArgumentParser(
    description='Run a LC fitter on a set of LC curves.')

parser = OptionParser()
# parser for fit parameters : 'dynamical' generation
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

covmb = None
mbCalc = yaml_params['mbcov']['estimate']

if mbCalc:
    #for this we need to have the SALT2 dir and files
    # if it does not exist get it from the web
    
    salt2Dir = yaml_params['mbcov']['directory']
    webPath = yaml_params['WebPath']
    check_get_dir(webPath,salt2Dir, salt2Dir)
    covmb = MbCov(salt2Dir, paramNames=dict(
        zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))

# create outputdir if necessary
outDir = yaml_params['Output']['directory']
if not os.path.isdir(outDir):
    os.makedirs(outDir)

prodid = yaml_params['ProductionID']
yaml_name = '{}/Fit_{}.yaml'.format(outDir, prodid)
with open(yaml_name, 'w') as f:
    data = yaml.dump(yaml_params, f)

# now fit all this
Fit_Simu(yaml_params, covmb)
