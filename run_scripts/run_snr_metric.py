import matplotlib.pyplot as plt
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.db as db
import argparse
import time
import yaml
from importlib import import_module
# import sqlite3
import numpy as np
from sn_tools.sn_cadence_tools import ReferenceData
import healpy as hp
import numpy.lib.recfunctions as rf
import sn_plotters.sn_snrPlotters as sn_plot

parser = argparse.ArgumentParser(
    description='Run a SN metric from a configuration file')
parser.add_argument('config_filename',
                    help='Configuration file in YAML format.')


def run(config_filename):
    # YAML input file.
    config = yaml.load(open(config_filename))
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
    # version = opsimdb.opsimVersion
    propinfo, proptags = opsimdb.fetchPropInfo()
    print('proptags and propinfo', proptags, propinfo)

    # grab the fieldtype (DD or WFD) from yaml input file
    fieldtype = config['Observations']['fieldtype']
    fake_file = config['Fake_file']
    module = import_module(config['Metric'])

    slicer = slicers.HealpixSlicer(nside=config['Pixelisation']['nside'])

    sqlconstraint = opsimdb.createSQLWhere(fieldtype, proptags)

    bundles = []
    names = []
    lim_sn = {}
    bands = config['Observations']['bands']
    z = config['Observations']['z']
    metric = {}
    # processing. Band after band

    Ra_ref = 0.000
    Dec_ref = -2.308039
    time_ref = time.time()
    for band in bands:
        sql_i = sqlconstraint+' AND '
        sql_i += 'filter = "%s"' % (band)
        # sql_i += ' AND abs(fieldRA-(%f))< %f' % (Ra_ref, 1.e-2)+' AND '
        # sql_i += 'abs(fieldDec-(%f))< %f' % (Dec_ref, 1.e-2)

        lim_sn[band] = ReferenceData(
            config['Li file'], config['Mag_to_flux file'], band, z)

        metric[band] = module.SNSNRMetric(lim_sn=lim_sn[band], names_ref=config['names_ref'], fake_file=fake_file, coadd=config['Observations']
                                          ['coadd'], z=z)
        bundles.append(metricBundles.MetricBundle(metric[band], slicer, sql_i))
        names.append(band)

    bdict = dict(zip(names, bundles))

    resultsDb = db.ResultsDb(outDir='None')
    mbg = metricBundles.MetricBundleGroup(bdict, opsimdb,
                                          outDir=outDir, resultsDb=resultsDb)

    mbg.runAll()

    # Let us display the results

    for band, val in bdict.items():
        metValues = val.metricValues[~val.metricValues.mask]
        res = None
        for vals in metValues:
            if res is None:
                res = vals
            else:
                res = np.concatenate((res, vals))
        res = np.unique(res)

        sn_plot.detecFracPlot(res, config['Pixelisation']
                              ['nside'], config['names_ref'])

        sn_plot.detecFracHist(res, config['names_ref'])

    plt.show()


def main(args):
    print('running')
    time_ref = time.time()
    run(args.config_filename)
    print('Time', time.time()-time_ref)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
