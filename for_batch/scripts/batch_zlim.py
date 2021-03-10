import pandas as pd
from optparse import OptionParser
import os
import numpy as np

class batch:
    def __init__(self, outputDir, config, ibatch, nproc):
    
        self.nproccomp = nproc
        dirScript, name_id, log = self.prepareOut(ibatch)
        
        self.script(dirScript, name_id, log, config, outputDir)


    def prepareOut(self,ibatch):
        """
        Method to prepare for the batch
    
        directories for scripts and log files are defined here.

        """

        self.cwd = os.getcwd()
        dirScript = self.cwd + "/scripts"

        if not os.path.isdir(dirScript):
            os.makedirs(dirScript)

        dirLog = self.cwd + "/logs"
        if not os.path.isdir(dirLog):
            os.makedirs(dirLog)

        name_id = 'zlim_{}'.format(ibatch)
        log = dirLog + '/'+name_id+'.log'

        return dirScript, name_id, log

    def script(self, dirScript, name_id, log, config, outputDir):
        """
        Method to generate and run the script to be executed

        Parameters
        ----------------
        dirScript: str
          location directory of the script
        name_id: str
          id for the script
        log: str
          location directory for the log files
        config: csv file
          parameters fot the processing
        outputDir: str
          output directory for data

        """
        # qsub command
        qsub = 'qsub -P P_lsst -l sps=1,ct=03:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
            log, self.nproccomp)

        scriptName = dirScript+'/'+name_id+'.sh'

        # fill the script
        script = open(scriptName, "w")
        script.write(qsub + "\n")
        script.write("#!/bin/env bash\n")
        script.write(" cd " + self.cwd + "\n")
        script.write(" echo 'sourcing setups' \n")
        script.write(" source setup_release.sh Linux\n")
        script.write("echo 'sourcing done' \n")

        cmd = 'python run_script/fakes/gensimufit'
        cmd += ' --mbcov_estimate 1'
        cmd += ' --outputDir {}'.format(outputDir)
        cmd += ' --config {}'.format(config)

        script.write(cmd + '\n')
        script.write("EOF" + "\n")
        script.close()
        #os.system("sh "+scriptName)



parser = OptionParser()

parser.add_option("--fileName", type="str", default='config.csv',
                  help="file to process [%default]")
parser.add_option("--n_per_file", type=int, default=10,
                  help="number of conf to process per batch [%default]")
parser.add_option("--dirConfig", type='str', default='configs_zlim',
                  help="where the config files will be located [%default]")
parser.add_option("--outputDir", type='str', default='/sps/lsst/users/gris/zlim_studies',
                  help="where output files will be located [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of cores for multiprocessing [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

fileName = opts.fileName
n_per_file = opts.n_per_file
dirConfig = opts.dirConfig
outputDir = opts.outputDir
nproc = opts.nproc

#create config dir if necessary
if not os.path.exists(dirConfig):
    os.mkdir(dirConfig)

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

df = pd.read_csv(fileName)

nz = len(df)
nproc = int(nz/n_per_file)
t = np.linspace(0, nz, nproc+1, dtype='int')
for j in range(nproc):
    print(t[j],t[j+1])
    outName = fileName.replace('.csv','_{}.csv'.format(j))
    newName = '{}/{}'.format(dirConfig,outName)
    df[t[j]:t[j+1]].to_csv(newName, index=False)
    batch(outputDir,newName,j,nproc)
    break
