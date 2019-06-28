import matplotlib.pyplot as plt
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import argparse
import time
import yaml
from importlib import import_module
import sqlite3
import numpy as np
from sn_tools.sn_cadence_tools import GenerateFakeObservations
import os
from sn_tools.sn_utils import GetReference

parser = argparse.ArgumentParser(
    description='Run a SN metric from a configuration file')
parser.add_argument('config_filename',
                    help='Configuration file in YAML format.')


def run(config_filename):
    # YAML input file.
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    # print(config)
    outDir = 'Test'  # this is for MAF

    # grab the db filename from yaml input file
    dbFile = config['Observations']['filename']

    """
    conn = sqlite3.connect(dbFile)
    cur = conn.cursor()
    table_name='Proposal'
    result = cur.execute("PRAGMA table_info('%s')" % table_name).fetchall()
    print('Results',result)
    cur.execute("SELECT * FROM Proposal")
    rows = cur.fetchall()
    for row in rows:
        print(row)
    print('end')
    cur.execute('PRAGMA TABLE_INFO({})'.format('ObsHistory'))
    
    names = [tup[1] for tup in cur.fetchall()]
    print(names)
    """

    #check whether X0_norm file exist or not (and generate it if necessary)
    absMag = config['SN parameters']['absmag']
    salt2Dir = config['SN parameters']['salt2Dir']
    model = config['Simulator']['model']
    version = str(config['Simulator']['version'])

    x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

    if not os.path.isfile(x0normFile):
        from sn_tools.sn_utils import X0_norm
        X0_norm(salt2Dir=salt2Dir,model=model, version=version, absmag=absMag,outfile=x0normFile)

    x0_tab = np.load(x0normFile)

    reference_lc = None
    if 'sn_fast' in config['Simulator']['name']:
        print('Loading reference LCs from',config['Simulator']['Reference File'])
        reference_lc = GetReference(
            config['Simulator']['Reference File'],config['Instrument'])
        print('Reference LCs loaded')



    module = import_module(config['Metric'])

    if dbFile != 'None':
        if dbFile.endswith('.db'):
            opsimdb = db.OpsimDatabase(dbFile)
            version = opsimdb.opsimVersion
            propinfo, proptags = opsimdb.fetchPropInfo()
            print('proptags and propinfo', proptags, propinfo)

            # grab the fieldtype (DD or WFD) from yaml input file
            fieldtype = config['Observations']['fieldtype']
            slicer = slicers.HealpixSlicer(nside=config['Pixelisation']['nside'])

            # print('slicer',slicer.pixArea,slicer.slicePoints['ra'])
            #print('alors condif', config)
            metric = module.SNMetric(
                config=config, coadd=config['Observations']['coadd'],x0_norm=x0_tab,reference_lc=reference_lc)

            sqlconstraint = opsimdb.createSQLWhere(fieldtype, proptags)

            mb = metricBundles.MetricBundle(metric, slicer, sqlconstraint)

            mbD = {0: mb}
            
            resultsDb = db.ResultsDb(outDir=outDir)
            
            mbg = metricBundles.MetricBundleGroup(mbD, opsimdb,
                                              outDir=outDir, resultsDb=resultsDb)
            
            mbg.runAll()
            

        if dbFile.endswith('.npy'):
            metric = module.SNMetric(
                config=config, coadd=False,x0_norm=x0_tab,reference_lc=reference_lc)
            
            observations = np.load(dbFile)

            metric.run(observations)

        if metric.save_status:
            metric.simu.Finish()

    else:
        config_fake = yaml.load(open(config['Param_file']), Loader=yaml.FullLoader)
        fake_obs = GenerateFakeObservations(config_fake).Observations

        metric = module.SNMetric(config=config,x0_norm=x0_tab)
        metric.run(fake_obs)
       
    # mbg.plotAll(closefigs=False)
    # plt.show()


def main(args):
    print('running')
    time_ref = time.time()
    run(args.config_filename)
    print('Time', time.time()-time_ref)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
