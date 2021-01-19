import numpy as np
from sn_tools.sn_io import loopStack
from scipy.interpolate import interp1d
import pandas as pd
from optparse import OptionParser
import os
import multiprocessing

def zlimit(tab, sigmaC=0.04):

    print('there tab',tab)
    interp = interp1d(np.sqrt(tab['Cov_colorcolor']), tab['z'], bounds_error=False, fill_value=0.)

    return np.round(interp(sigmaC),2)

def plot(tab, sigmaC=0.04):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(tab['z'],np.sqrt(tab['Cov_colorcolor']))

    ax.plot(ax.get_xlim(), [sigmaC]*2,color='r')
    ax.grid()
    ax.set_xlabel('z')
    ax.set_ylabel('$\sigma_{C}$')
    plt.show()

def ana_zlim(case,simulator='sn_fast',fitter='sn_fast',x1=-2.0,color=0.2,ebvofMW=0.0,snrmin=1.,error_model=1,bluecutoff=380.,redcutoff=800.):

    cutoff = 'error_model'
    if error_model < 1:
        cutoff = '{}_{}'.format(bluecutoff,redcutoff)
    
    dirFile = 'Output_Fit_{}_ebvofMW_{}_snrmin_{}'.format(cutoff,ebvofMW,int(snrmin))

    fName = 'Fit_{}_Fake_{}_{}_{}_ebvofMW_{}_{}_{}.hdf5'.format(simulator,x1,color,cutoff,ebvofMW,case,fitter)

    fullName = '{}/{}'.format(dirFile,fName)

    print('analyzing',fullName)
    tab = loopStack([fullName],'astropyTable')

    idx = tab['z']>=0.1
    sel = tab[idx]

    #print(sel.columns,zlim(sel))
    #plot(sel)

    return zlimit(sel)

def simufit_cmd(x1,color, ebvofMW,simulator, fitter,error_model,tagprod,Nvisits,m5,cadence,bands):

    cmd = 'python run_scripts/fakes/loop_full_fast.py'
    cmd += ' --x1 {}'.format(x1)
    cmd += ' --color {}'.format(color)
    cmd += ' --ebv {}'.format(ebvofMW)
    cmd += ' --simus {}'.format(simulator)
    cmd += ' --{}_simu_fitter {}'.format(simulator, fitter)
    cmd += ' --error_model {}'.format(error_model)
    cmd += ' --tagprod {}'.format(tagprod)

    for b in bands:
        cmd += ' --Nvisits_{} {}'.format(b,Nvisits[b])
        cmd += ' --m5_{} {}'.format(b,m5[b])
        cmd += ' --cadence_{} {}'.format(b,cadence[b])

    return cmd

def multiproc(conf, bands,nproc=8):

    nz = len(conf)
    t = np.linspace(0, nz, nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=process,
                                     args=(conf[t[j]:t[j+1]], bands, j, result_queue))
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
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals), sort=False)
        """
            if restot is None:
                restot = vals
            else:
                restot = np.concatenate((restot, vals))
        """
        return restot
    

def process(config,bands, j=0, output_q=None):
    rz = []
    confres = pd.DataFrame(config)
    for index, conf in config.iterrows():
        # first step: simulate and fit
        Nvisits = {}
        m5 = {}
        cadence = {}

        for b in bands:
            Nvisits[b] = conf['N{}'.format(b)]
            m5[b] = conf['m5_{}'.format(b)]
            cadence[b] = conf['cadence_{}'.format(b)]

        cmd = simufit_cmd(conf['x1'],conf['color'], conf['ebvofMW'],
                          conf['simulator'], conf['fitter'],conf['error_model'],conf['tagprod'],Nvisits,m5,cadence,bands)
        print(cmd)
        os.system(cmd)
    
   
        zlim = ana_zlim(conf['tagprod'],
                        simulator=conf['simulator'],
                        fitter=conf['fitter'],
                        x1=conf['x1'],
                        color=conf['color'],
                        ebvofMW=conf['ebvofMW'],
                        snrmin=conf['snrmin'],
                        error_model=conf['error_model'],
                        bluecutoff=conf['bluecutoff'],
                        redcutoff=conf['redcutoff'])
        print(zlim)
        rz.append(zlim)

    confres['zlim'] = rz

    if output_q is not None:
        return output_q.put({j: confres})
    else:
        return confres
parser = OptionParser()

parser.add_option("--config", type='str', default='run_scripts/fakes/config.csv',
                  help="config file to use[%default]")

opts, args = parser.parse_args()

config= pd.read_csv(opts.config,comment='#')

bands = 'grizy'
res = multiproc(config, bands,nproc=1)



