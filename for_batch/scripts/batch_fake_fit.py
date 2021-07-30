from optparse import OptionParser
import pandas as pd
import os
import glob
from sn_tools.sn_io import loopStack
import numpy as np
import multiprocessing


def process(dbName, prodid, simuDir, outDir, num, nproc=8, mode='batch', snrmin=5., tag='gg', mbcov=0):

    if mode == 'batch':
        batch(dbName, prodid, simuDir,
              outDir, num, nproc, snrmin, tag, mbcov)
    else:
        if prodid:
            cmd_ = cmd(dbName, prodid.item(), simuDir, outDir, nproc, snrmin)
            # os.system(cmd_)
            print('will execute', prodid.item())
            # os.system(cmd_)


def batch(dbName, prodids, simuDir, outDir, num, nproc=8, snrmin=5., tag='gg', mbcov=0):

    dirScript, name_id, log, cwd = prepareOut(dbName, num, tag)
    # qsub command
    qsub = 'qsub -P P_lsst -l sps=1,ct=3:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
        log, nproc)

    scriptName = dirScript+'/'+name_id+'.sh'

    # fill the script
    script = open(scriptName, "w")
    script.write(qsub + "\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux\n")
    script.write("echo 'sourcing done' \n")

    # need this to limit the number of multithread
    script.write(" export MKL_NUM_THREADS=1 \n")
    script.write(" export NUMEXPR_NUM_THREADS=1 \n")
    script.write(" export OMP_NUM_THREADS=1 \n")
    script.write(" export OPENBLAS_NUM_THREADS=1 \n")

    for prodid in prodids:
        cmd_ = cmd(dbName, prodid, simuDir, outDir, nproc, snrmin, mbcov)
        script.write(cmd_+" \n")

    script.write("EOF" + "\n")
    script.close()
    #os.system("sh "+scriptName)


def prepareOut(dbName, num, tag):
    """                                                                                        
    Function to prepare for the batch                                                            

    directories for scripts and log files are defined here.                                    

    """

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}_{}_{}'.format(dbName, num, tag)

    name_id = 'fit_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    return dirScript, name_id, log, cwd


def cmd(dbName, prodid, simuDir, outDir, nproc, snrmin, mbcov):
    cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
    #cmd += ' --ProductionID {}_{}_allSN_{}_sn_cosmo'.format(dbName,fieldName,num)
    #cmd += ' --Simulations_prodid {}_{}_allSN_{}'.format(dbName,fieldName,num)
    cmd += ' --ProductionIDFit {}_sn_cosmo'.format(prodid)
    cmd += ' --Simulations_prodid {}'.format(prodid)
    cmd += ' --Simulations_dirname {}'.format(simuDir)
    cmd += ' --LCSelection_snrmin {}'.format(snrmin)
    cmd += ' --LCSelection_nbands 0'
    cmd += ' --OutputFit_directory {}/{}'.format(outDir, dbName)
    cmd += ' --MultiprocessingFit_nproc {}'.format(nproc)
    cmd += ' --mbcov_estimate {}'.format(mbcov)
    return cmd


def nlc(simuFile):
    """
    Method estimating the number of LC

    Parameters
    ----------
    simuFile: str
      name of the simuFile

    Returns
    -------
    number of LCs

    """

    res = loopStack(simuFile, objtype='astropyTable')

    return len(res)


def family(dbName, rlist):
    """
    Method to get a family from a dbName

    Parameters
    ----------
    dbName: str
    the dbName to process

    Returns
    ----------
    str: the name of the 'family'

    """

    ro = []
    fam = dbName
    for i in range(len(dbName)):
        stre = dbName[:i+1]
        num = 0
        for kk in rlist:
            if stre == kk[:i+1]:
                num += 1
        # print(stre, num)
        ro.append(num)
        if i > 5 and ro[-1]-ro[-2] < 0:
            fam = dbName[:i]
            break

    return fam

# get the simufile here


def getSimu_multi(simuDir, dbName, snType, nproc):

    fullsimuDir = '{}/{}'.format(simuDir, dbName)
    simuFiles = glob.glob(
        '{}/Simu*{}*.hdf5'.format(fullsimuDir, snType))

    nz = len(simuFiles)
    nproc = min([nz, nproc])
    print('nproc')
    # multiprocessing parameters
    t = np.linspace(0, nz, nproc+1, dtype='int')
    # print('multi', nz, t)
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=getSimu,
                                     args=(simuFiles[t[j]:t[j+1]], j, result_queue))
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
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals))

    return restot


def getSimu(simuFiles, j=0, output_q=None):

    # print('hh',simuFiles,len(simuFiles))

    simudf = pd.DataFrame(simuFiles, columns=['simuFile'])

    simudf['prodid'] = simudf['simuFile'].apply(
        lambda x: x.split('/')[-1].split('.hdf5')[0].split('Simu_')[-1])

    # print('ggg',j,simudf['simuFile'].unique())
    simudf = simudf.groupby(['prodid']).apply(
        lambda x: pd.DataFrame({'nlc': [nlc(x['simuFile'])]})).reset_index()

    # print('rrr',simudf)
    if output_q is not None:
        return output_q.put({j: simudf})
    else:
        return simudf


parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.5_10yrs',
                  help="dbName to process  [%default]")
parser.add_option("--simuDir", type="str",
                  default='/sps/lsst/users/gris/DD/Simu', help="simu dir [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/DD/Fit', help="output directory [%default]")
parser.add_option("--mode", type="str", default='batch',
                  help="run mode batch/interactive[%default]")
parser.add_option("--snrmin", type=float, default=1.,
                  help="min snr for LC points fit[%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of proc to use[%default]")
parser.add_option("--snTypes", type='str', default='faintSN,allSN',
                  help="tag for production [%default]")
parser.add_option("--mbcov_estimate", type=int, default=0,
                  help="to estimate covmb[%default]")

opts, args = parser.parse_args()

print('Start processing...')


nlc_ref = 5000
#nlc_ref = 2000
snTypes = opts.snTypes.split(',')

#snType = ['faintSN','allSN']
#snType = ['mediumSN']
simuDir = '{}/{}'.format(opts.simuDir, opts.dbName)
for bb in snTypes:
    simudf = getSimu_multi(opts.simuDir, opts.dbName, bb, opts.nproc)
    print(bb, simudf[['prodid', 'nlc']], len(simudf))

    list_proc = []
    nlc_sim = 0
    iproc = 0
    for ic, row in simudf.iterrows():
        #process(opts.dbName,opts.fieldName,[row['prodid']], simuDir, opts.outDir,ic,opts.nproc,opts.mode,opts.snrmin,bb)
        nlc_sim += row['nlc']
        list_proc += [row['prodid']]

        if nlc_sim >= nlc_ref:
            iproc += 1
            print('processing', iproc, list_proc, nlc_sim)
            process(opts.dbName, list_proc, simuDir, opts.outDir, iproc,
                    opts.nproc, opts.mode, opts.snrmin, bb, opts.mbcov_estimate)
            list_proc = []
            nlc_sim = 0

    if nlc_sim > 0:
        iproc += 1
        process(opts.dbName, list_proc, simuDir, opts.outDir, iproc,
                opts.nproc, opts.mode, opts.snrmin, bb, opts.mbcov_estimate)

    print('total number of sn', nlc_sim)
"""
print(test)

for i in range(8):
    for bb in snType:                                                                                            
        simudf = getSimu(opts.simuDir,opts.dbName,opts.fieldName, bb)
        idx = simudf['prodid'].str.contains('{}_{}'.format(bb,i))
        sel = simudf[idx].to_records()
        if len(sel) >0:
            nlc = np.sum(sel['nlc'])
            print('oo',bb,i,ic,nlc)
            if nlc/nlc_ref >=1.1:
               nn = int(nlc/nlc_ref)
               bbatch = np.linspace(0, len(sel), nn+1, dtype='int')
               print(bbatch,len(sel))
               kk = 0
               for ibb in range(len(bbatch)-1):
                   ic += 1
                   ia = bbatch[ibb]
                   ib = bbatch[ibb+1]
                   #print('bb',ia,ib,sel[ia:ib]['nlc'].sum(),sel[ia:ib]['prodid'])
                   kk += sel[ia:ib]['nlc'].sum()
                   process(opts.dbName,opts.fieldName,sel[ia:ib]['prodid'], opts.simuDir, opts.outDir,ic,opts.nproc,opts.mode,opts.snrmin)
               print('finally',kk)
               
            else:
                ic +=1
                process(opts.dbName,opts.fieldName,sel['prodid'].tolist(), opts.simuDir, opts.outDir,ic,opts.nproc,opts.mode,opts.snrmin)

"""
"""
dbName = 'descddf_v1.4_10yrs'
fieldName = 'COSMOS'


for num in range(8):
    cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
    cmd += ' --ProductionID {}_{}_allSN_{}_sn_cosmo'.format(dbName,fieldName,num) 
    cmd += ' --Simulations_prodid {}_{}_allSN_{}'.format(dbName,fieldName,num)
    cmd += ' --Simulations_dirname /sps/lsst/users/gris/DD/Simu/{}'.format(dbName)
    cmd += ' --LCSelection_snrmin 5.' 
    cmd += ' --LCSelection_nbands 3' 
    cmd += ' --Output_directory /sps/lsst/users/gris/DD/Fit' 
    cmd += ' --Multiprocessing_nproc 8'
    print(cmd)
    os.system(cmd)
    
"""
