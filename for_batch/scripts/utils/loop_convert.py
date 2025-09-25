import pandas as pd
import os

csvFile = 'DD_fbs_5.0.0.csv'
tt = pd.read_csv(csvFile,comment='#')

print(tt)

script = 'python run_scripts/utils/convert_db_to_npy.py'
for i, row in tt.iterrows():
    dbName = row['dbName']
    inputDir = row['dbDir']
    outputDir = inputDir.replace('/db','/npy')
    dbNamepy = '{}.npy'.format(dbName)
    fName = '{}/{}'.format(outputDir,dbNamepy)
    print(fName)
    if not os.path.isfile(fName):
        scr_ = '{}'.format(script)
        scr_ += ' --inputDir={}'.format(inputDir)
        scr_ += ' --outputDir={}'.format(outputDir)
        scr_ += ' --dbName={}'.format(dbName)
        print(scr_)
        os.system(scr_)
