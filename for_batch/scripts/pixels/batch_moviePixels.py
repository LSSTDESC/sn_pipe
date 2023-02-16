import pandas as pd
from optparse import OptionParser
import numpy as np
import os

def batch(dbNames,id,script,nightmin,nightmax,dbDir,dbDir_pixels,figDir,movieDir,ffmpeg,nproc=8):
 

    cmd_process = ' python {}'.format(script)
    cmd_process += ' --dispType moviepixels --saveMovie 1'
    cmd_process += ' --nights {}-{}'.format(nightmin,nightmax)
    cmd_process += ' --dbDir {} --dbDir_pixels {} --figDir {}  --movieDir {}'.format(dbDir,dbDir_pixels,figDir,movieDir) 
    cmd_process += ' --ffmpeg {}'.format(ffmpeg)
    cmd_process += ' --mode batch'
    

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"
    
    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)


    name_id = 'moviepixels_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'
    
    # qsub command
    qsub = 'qsub -P P_lsst -l sps=1,ct=12:00:00,h_vmem=20G -j y -o {} -pe multicores {} <<EOF'.format(
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

    #print('hhh',dbNames)
    #for vv in dbNames:
    dbSplit = np.array_split(dbNames,3)

    for vvo in dbSplit:
        dbName = ','.join(vvo)
        cmd_ = '{} --dbName {}'.format(cmd_process,dbName)
   
        script.write(cmd_+" \n")
    
    script.write("EOF" + "\n")
    script.close()
    os.system("sh "+scriptName)


parser = OptionParser()

parser.add_option("--dbList", type="str", default='WFD_fbs16.csv',
                  help="dbList to process  [%default]")
parser.add_option("--dbDir", type="str", default='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db',
                  help="db dir [%default]")
parser.add_option("--dbDir_pixels", type="str", default='/sps/lsst/users/gris/ObsPixelized_circular_new',
                  help="db pixel dir [%default]")
parser.add_option("--figDir", type="str", default='/sps/lsst/users/gris/OS_Figures',
                  help="fig dir [%default]")
parser.add_option("--movieDir", type="str", default='/sps/lsst/users/gris/web/OS_Videos/Gaps',
                  help="movie dir [%default]")
parser.add_option("--nightmin", type=int, default=1,
                  help="first night to consider[%default]")
parser.add_option("--nightmax", type=int, default=730,
                  help="lastnight to consider[%default]")
parser.add_option("--ffmpeg", type=str, default='../../ffmpeg/ffmpeg-4.3.1-i686-static/ffmpeg',
                  help="ffmpeg command [%default]")
parser.add_option("--version", type=str, default='fbs_1.6',
                  help="simu version [%default]")

"""
    cmd = ' python run_scripts/visu_cadence/run_visu_cadence.py --dispType moviepixels --nights 1-730 --saveMovie 1 --dbDir_pixels /sps/lsst/users/gris/ObsPixelized_circular_new --dbDir /sps/lsst/cadence/LSST_SN_CADENCE/cadence_db --figDir /sps/lsst/users/gris/OS_Figures --movieDir /sps/lsst/users/gris/web/OS_Videos/Gaps --ffmpeg ../../ffmpeg/ffmpeg-4.3.1-i686-static/ffmpeg --dbName'
"""

opts, args = parser.parse_args()

dbList = pd.read_csv(opts.dbList, comment='#')

print(dbList,len(dbList))

dfs = np.array_split(dbList['dbName'].to_list(),5)

for dd in dfs:
    print(dd.shape)

for iu,dd in enumerate(dfs):
    if len(dd) >0:
        print('processing',dd)
        batch(dd,iu,
              script='run_scripts/visu_cadence/run_visu_cadence.py',
              nightmin=opts.nightmin,nightmax=opts.nightmax,
              dbDir=opts.dbDir,dbDir_pixels=opts.dbDir_pixels,
              figDir=opts.figDir,movieDir='{}/{}'.format(opts.movieDir,opts.version),
              ffmpeg=opts.ffmpeg)

"""
 cmd = ' python run_scripts/visu_cadence/run_visu_cadence.py --dispType moviepixels --nights 1-730 --saveMovie 1 --dbDir_pixels /sps/lsst/users/gris/ObsPixelized_circular_new --dbDir /sps/lsst/cadence/LSST_SN_CADENCE/cadence_db --figDir /sps/lsst/users/gris/OS_Figures --movieDir /sps/lsst/users/gris/web/OS_Videos/Gaps --ffmpeg ../../ffmpeg/ffmpeg-4.3.1-i686-static/ffmpeg --dbName'
"""
