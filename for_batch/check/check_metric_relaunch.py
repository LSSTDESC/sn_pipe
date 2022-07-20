import pandas as pd
import glob
import re
import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--cvs_file", type="str", default='WFD_fbs_2.0.csv',
                  help="list of db names to check [%default]")
parser.add_option("--logDir", type="str",
                  default='logs', help="log dir [%default]")
parser.add_option("--scriptDir", type="str",
                  default='scripts', help="script dir [%default]")
parser.add_option("--metricName", type="str",
                  default='NSNY', help="metric name [%default]")
parser.add_option("--fieldType", type="str",
                  default='WFD', help="field type [%default]")
parser.add_option("--nside", type=int,
                  default=64, help="nside [%default]")
parser.add_option("--nprocess", type=int,
                  default=8, help="expected number of process [%default]")

opts, args = parser.parse_args()

cvs_file = opts.cvs_file
logDir = opts.logDir
scriptDir = opts.scriptDir
metricName =opts.metricName
nside = opts.nside
fieldType = opts.fieldType
nprocess = opts.nprocess

# load cvs file

df = pd.read_csv(cvs_file)

#loop on dbNames

patrn = 'end of processing for'
for i, row in df.iterrows():
    
    dbName = row['dbName']
    search_path = '{}/metric_{}_{}_{}_{}*.log'.format(logDir,dbName,nside,fieldType,metricName)
    fis = glob.glob(search_path)
            
    for fi in fis:
        file_one = open(fi, "r")
        count = 0
        for word in file_one:
            if re.search(patrn, word):
                count += 1
        if count < nprocess:
            script = fi.replace('{}/'.format(logDir),'{}/'.format(scriptDir)).replace('.log','.sh')
            print('reprocessing needed here', dbName,script)
            cmd = 'sbatch {}'.format(script)
            os.system(cmd)
