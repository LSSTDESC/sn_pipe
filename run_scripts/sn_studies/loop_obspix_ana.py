import pandas as pd
import os

ll = pd.read_csv('for_batch/input/List_Db_DD.csv',comment='#')

for i, row in ll.iterrows():
    cmd = 'python sn_studies/sn_DD_nsn/obspixel_analysis_DD.py'
    cmd += ' --fileDir /sps/lsst/users/gris/ObsPixelized_128' 
    cmd += ' --outputDir /sps/lsst/users/gris/pixels_analysis'
    cmd += ' --dbName {}'.format(row['dbName'])
    os.system(cmd)

