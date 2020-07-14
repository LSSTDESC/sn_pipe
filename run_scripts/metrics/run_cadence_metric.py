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
import sn_plotters.sn_cadencePlotters as sn_plot

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
    opsimdb = db.OpsimDatabase(dbFile)
    version = opsimdb.opsimVersion
    propinfo, proptags = opsimdb.fetchPropInfo()
    print('proptags and propinfo', proptags, propinfo)

    # grab the fieldtype (DD or WFD) from yaml input file
    fieldtype = config['Observations']['fieldtype']

    module = import_module(config['Metric'])

    slicer = slicers.HealpixSlicer(nside=config['Pixelisation']['nside'])

    sqlconstraint = opsimdb.createSQLWhere(fieldtype, proptags)

    bundles = []
    names = []
    SNR = dict(zip(config['Observations']['bands'],
                   config['Observations']['SNR']))
    mag_range = config['Observations']['mag_range']
    dt_range = config['Observations']['dt_range']
    for band in SNR.keys():
        sql_i = sqlconstraint+' AND '
        sql_i += 'filter = "%s"' % (band)
        # sql_i += ' AND '
        # sql_i +=  'season= "%s"' % (season)
        metric = module.SNCadenceMetric(
            config=config, coadd=config['Observations']['coadd'])
        bundles.append(metricBundles.MetricBundle(metric, slicer, sql_i))
        names.append(band)

        print('sql', sql_i)

    bdict = dict(zip(names, bundles))

    resultsDb = db.ResultsDb(outDir='None')
    mbg = metricBundles.MetricBundleGroup(bdict, opsimdb,
                                          outDir=outDir, resultsDb=resultsDb)

    result = mbg.runAll()

    # Let us display the results

    for band, val in bdict.items():
        metValues = val.metricValues[~val.metricValues.mask],
        res = None
        for val in metValues:
            for vval in val:
                if res is None:
                    res = vval
                else:
                    res = np.concatenate((res, vval))
        res = np.unique(res)
        sn_plot.plotCadence(band, config['Li file'], config['Mag_to_flux file'],
                            SNR[band],
                            res,
                            config['names_ref'],
                            mag_range=mag_range, dt_range=dt_range)

    # mbg.writeAll()
    # mbg.plotAll(closefigs=False)
    # mbg.plot()
    plt.show()


def main(args):
    print('running')
    time_ref = time.time()
    run(args.config_filename)
    print('Time', time.time()-time_ref)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
