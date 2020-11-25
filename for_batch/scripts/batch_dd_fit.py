from optparse import OptionParser
import pandas as pd
import os
import glob
from sn_tools.sn_io import loopStack
import numpy as np

def process(dbName,fieldName,prodid, simuDir, outDir,num,nproc=8,mode='batch',snrmin=5.):

    if mode == 'batch':
        batch(dbName,fieldName,prodid, simuDir, outDir,num,nproc,snrmin)
    else:
        cmd_ = cmd(dbName,prodid,simuDir,outDir,nproc,snrmin)
        os.system(cmd_)

def batch(dbName,fieldName,prodids, simuDir, outDir,num,nproc=8,snrmin=5.):

    dirScript, name_id, log, cwd = prepareOut(dbName,fieldName,num)
    # qsub command                                                                             
    qsub = 'qsub -P P_lsst -l sps=1,ct=3:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log, nproc)

    scriptName = dirScript+'/'+name_id+'.sh'

    # fill the script                                                                          
    script = open(scriptName, "w")
    script.write(qsub + "\n")
    script.write("#!/bin/env bash\n")
    script.write(" cd " + cwd + "\n")
    script.write(" echo 'sourcing setups' \n")
    script.write(" source setup_release.sh Linux\n")
    script.write("echo 'sourcing done' \n")

   
    for prodid in prodids:
        cmd_=cmd(dbName,prodid, simuDir, outDir,nproc,snrmin)
        script.write(cmd_+" \n")

    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)
    

def prepareOut(dbName,fieldName,num):
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

    id = '{}_{}_{}'.format(dbName, fieldName,num)

    name_id = 'fit_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'

    return dirScript, name_id, log, cwd


def cmd(dbName,prodid,simuDir,outDir,nproc,snrmin):
    cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
    #cmd += ' --ProductionID {}_{}_allSN_{}_sn_cosmo'.format(dbName,fieldName,num) 
    #cmd += ' --Simulations_prodid {}_{}_allSN_{}'.format(dbName,fieldName,num)
    cmd += ' --ProductionID {}_sn_cosmo'.format(prodid) 
    cmd += ' --Simulations_prodid {}'.format(prodid)
    cmd += ' --Simulations_dirname {}'.format(simuDir)
    cmd += ' --LCSelection_snrmin {}'.format(snrmin) 
    cmd += ' --LCSelection_nbands 3' 
    cmd += ' --Output_directory {}/{}'.format(outDir,dbName) 
    cmd += ' --Multiprocessing_nproc {}'.format(nproc)
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
    res = loopStack(simuFile,objtype='astropyTable')

    return len(res)

def family(dbName,rlist):
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

parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',help="dbName to process  [%default]")
parser.add_option("--simuDir", type="str", default='/sps/lsst/users/gris/DD/Simu',help="simu dir [%default]")
parser.add_option("--fieldName", type="str", default='COSMOS',help="DD field to process [%default]")
parser.add_option("--outDir", type="str", default='/sps/lsst/users/gris/DD/Fit',help="output directory [%default]")
parser.add_option("--mode", type="str", default='batch',help="run mode batch/interactive[%default]")
parser.add_option("--snrmin", type=float, default=1.,help="min snr for LC points fit[%default]")
parser.add_option("--nproc", type=int, default=8,help="number of proc to use[%default]")

opts, args = parser.parse_args()

print('Start processing...')


#get the simufile here

simuDir='{}/{}'.format(opts.simuDir,opts.dbName)
simuFiles = glob.glob('{}/Simu*{}*.hdf5'.format(simuDir,opts.fieldName))

print('hh',simuFiles,len(simuFiles))

simudf = pd.DataFrame(simuFiles,columns=['simuFile'])

simudf['prodid'] = simudf['simuFile'].apply(lambda x : x.split('/')[-1].split('.hdf5')[0].split('Simu_')[-1])

simudf = simudf.groupby(['prodid']).apply(lambda x: pd.DataFrame({'nlc': [nlc(x['simuFile'])]})).reset_index()

print(simudf)

ic = -1
nlc_ref = 10000
for i in range(8):
    for bb in ['faintSN','allSN']:                                                                                            
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
                   process(opts.dbName,opts.fieldName,sel[ia:ib]['prodid'], simuDir, opts.outDir,ic,opts.nproc,opts.mode,opts.snrmin)
               print('finally',kk)
               
            else:
                ic +=1
                process(opts.dbName,opts.fieldName,sel['prodid'].tolist(), simuDir, opts.outDir,ic,opts.nproc,opts.mode,opts.snrmin)


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
