import pandas as pd
import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--OSlist', type=str, default='for_batch/input/List_Db_DD.csv',
                  help='OS list to process [%default]')
parser.add_option('--pixel_ana_dir', type=str, default='/sps/lsst/users/gris/pixels_analysis',
                  help='pixel analysis file directory [%default]')
parser.add_option('--action', type=str, default='pixels_analysis,OS_downtime',
                  help='pixel analysis file directory [%default]')
parser.add_option('--pixelDir', type=str, default='/sps/lsst/users/gris/ObsPixelized_128',
                  help='obs pixels directory [%default]')
parser.add_option('--outputDir', type=str, default='/sps/lsst/users/gris/OS_downtime',
                  help='output directory [%default]')

opts, args = parser.parse_args()

actions = opts.action.split(',')

ll = pd.read_csv(opts.OSlist, comment='#')

pixel_ana_dir = opts.pixel_ana_dir
for i, row in ll.iterrows():
    cmd = 'python sn_studies/sn_DD_nsn/obspixel_analysis_DD.py'
    cmd += ' --fileDir {}'.format(opts.pixelDir)
    cmd += ' --outputDir {}'.format(pixel_ana_dir)
    cmd += ' --dbName {}'.format(row['dbName'])
    cmd += ' --fieldNames COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'
    if 'pixels_analysis' in actions:
        os.system(cmd)
    cmd = 'python sn_studies/sn_DD_nsn/OS_downtime.py' 
    cmd += ' --dbDir {}'.format(row['dbDir'])
    cmd += ' --dbName {}'.format(row['dbName']) 
    cmd += ' --fieldNames COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb' 
    cmd += ' --fieldDir {}'.format(pixel_ana_dir)
    cmd += ' --outputDir {}'.format(opts.outputDir)
    if 'OS_downtime' in actions:
        os.system(cmd)
