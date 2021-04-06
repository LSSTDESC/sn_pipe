import os
from optparse import OptionParser
import pandas as pd

class Process:
    def __init__(self, dbDir, dbName,dbExtens,outDir,pixelmap_dir, nclusters, nside):
        
        self.dbDir =dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.outDir = outDir
        self.pixelmap_dir = pixelmap_dir
        self.nclusters = nclusters
        self.nside = opts.nside

    def __call__(self, nproc, fieldNames, ebvofMW,mode):

        if opts.mode == 'interactive':
            for fieldName in fieldNames:
                cmd_proc = self.cmd(nproc,fieldName,ebvofMW)
                print(cmd_proc)
        
        if opts.mode == 'batch':
            self.batch(nproc,fieldNames,ebvofMW)

            
    def cmd(self,nproc,fieldName,ebvofMW):

        cmd_ = 'python run_scripts/metrics/run_metrics.py'
        cmd_ += ' --dbName {}'.format(self.dbName)
        cmd_ += ' --dbExtens {}'.format(self.dbExtens)
        cmd_ += ' --dbDir {}'.format(self.dbDir)
        cmd_ += ' --fieldName {}'.format(fieldName)
        cmd_ += ' --fieldType DD'
        cmd_ += ' --zmax 1.0'
        cmd_ += ' --nproc {}'.format(nproc) 
        cmd_ += ' --metric NSN' 
        cmd_ += ' --nclusters {}'.format(self.nclusters)
        cmd_ += ' --ebvofMW {}'.format(ebvofMW) 
        cmd_ += ' --npixels -1'
        cmd_ += ' --saveData 1'
        cmd_ += ' --pixelmap_dir {}'.format(self.pixelmap_dir)
        cmd_ += ' --outDir {}'.format(self.outDir)
        cmd_ += ' --nside {}'.format(self.nside)

        return cmd_

    def batch(self,nproc,fieldNames,ebvofMW):

        dirScript, name_id, log, cwd = self.prepareOut()
        # qsub command                                                                             
        qsub = 'qsub -P P_lsst -l sps=1,ct=6:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(log, nproc)

        scriptName = dirScript+'/'+name_id+'.sh'

        # fill the script                                                                          
        script = open(scriptName, "w")
        script.write(qsub + "\n")
        script.write("#!/bin/env bash\n")
        script.write(" cd " + cwd + "\n")
        script.write(" echo 'sourcing setups' \n")
        script.write(" source setup_release.sh Linux\n")
        script.write("echo 'sourcing done' \n")

        for fieldName in fieldNames:
                cmd_proc = self.cmd(nproc,fieldName,ebvofMW)
                print(cmd_proc)
                script.write(cmd_proc+" \n")

        script.write("EOF" + "\n")
        script.close()
        os.system("sh "+scriptName)
        
    
    def prepareOut(self):
        """                                                                                        
        method to prepare for the batch                                                            
    
        directories for scripts and log files are defined here.                                    
                                                                                                   
        """

        cwd = os.getcwd()
        dirScript = cwd + "/scripts"

        if not os.path.isdir(dirScript):
            os.makedirs(dirScript)

        dirLog = cwd + "/logs"
        if not os.path.isdir(dirLog):
            os.makedirs(dirLog)

        id = '{}'.format(self.dbName)

        name_id = 'metric_DD_{}'.format(id)
        log = dirLog + '/'+name_id+'.log'

        return dirScript, name_id, log, cwd

    
parser = OptionParser()

parser.add_option('--dbList', type='str', default='for_batch/input/List_Db_DD.csv',help='list of dbNames to process  [%default]')
parser.add_option('--outDir', type='str', default='/sps/lsst/users/gris/MetricOutput_DD_new_128',help='output Dir to  [%default]')
parser.add_option('--mode', type='str', default='batch',help='running mode batch/interactive [%default]')
parser.add_option('--snrmin', type=float, default=1.,help='min snr for LC point fit[%default]')
parser.add_option('--pixelmap_dir', type='str', default='/sps/lsst/users/gris/ObsPixelized_128',help='pixelmap directory [%default]')
parser.add_option('--nproc', type=int, default=8,help='number of proc [%default]')
parser.add_option('--ebvofMW', type=float, default=-1.0,help='E(B-V) [%default]')
parser.add_option('--fieldNames', type=str, default='COSMOS,CDFS,ELAIS,XMM-LSS,ADFS1,ADFS2',help='DD fields to process [%default]')
parser.add_option('--nclusters', type=int, default=6,help='total number of DD in this OS [%default]')
parser.add_option('--nside', type=int, default=128,help='healpix nside parameter [%default]')

opts, args = parser.parse_args()

toprocess = pd.read_csv(opts.dbList, comment='#')

fieldNames = opts.fieldNames.split(',')

for i,row in toprocess.iterrows():

    proc = Process(row['dbDir'], row['dbName'],row['dbExtens'],opts.outDir,opts.pixelmap_dir, opts.nclusters,opts.nside)

    proc(opts.nproc,fieldNames, opts.ebvofMW,opts.mode)
