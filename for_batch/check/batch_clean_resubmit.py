#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:26:28 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import os

def get_jobid(fi):
    """
    Function to grab the Job id from log file

    Parameters
    ----------
    fi : str
        file name.

    Returns
    -------
    int
        jobid.

    """
    
    file_one = open(fi, "r")
       
    jobid_str = 'Job id:'
    for word in file_one:
        if jobid_str in word:
            jobid = word.split(jobid_str)[1]
            return int(jobid)

    return -1

parser = OptionParser(description='script to clean and relaunch batch jobs')

parser.add_option("--csv_file", type="str", default='files_err.csv',
                  help="list of files with errors [%default]")
parser.add_option("--logDir", type="str",
                  default='logs', help="log dir [%default]")
parser.add_option("--scriptDir", type="str",
                  default='scripts', help="script dir [%default]")

opts, args = parser.parse_args()

csv_file = opts.csv_file
logDir = opts.logDir
scriptDir = opts.scriptDir

#load files
df = pd.read_csv(csv_file)

for i, row in df.iterrows():
    #grab the log file
    fi = row['file']
    log_file = fi.replace('.err','.log')
    jobid = get_jobid(log_file)
    if jobid == -1:
        print('problem: Job id not found in log file!!!!!!')
    #cancel this job
    cmd = 'scancel {}'.format(jobid)
    print(cmd)
    os.system(cmd)
    script = fi.replace('{}/'.format(logDir),'{}/'.format(scriptDir)).replace('.err','.sh')
    cmd = 'sbatch {}'.format(script)
    print(cmd)
    os.system(cmd)