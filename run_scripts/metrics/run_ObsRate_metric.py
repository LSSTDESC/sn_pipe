import matplotlib.pyplot as plt
import rubin_sim.maf.metricBundles as metricBundles
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.db as db
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
    description='Run SN ObsRate metric from a configuration file')
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
    # version = opsimdb.opsimVersion
    propinfo, proptags = opsimdb.fetchPropInfo()
    print('proptags and propinfo', proptags, propinfo)

    # grab the fieldtype (DD or WFD) from yaml input file
    fieldtype = config['Observations']['fieldtype']
    season = config['Observations']['season']
    module = import_module(config['Metric'])

    slicer = slicers.HealpixSlicer(nside=config['Pixelisation']['nside'])

    sqlconstraint = opsimdb.createSQLWhere(fieldtype, proptags)

    bundles = []
    names = []
    lim_sn = {}
    bands = config['Observations']['bands']
    snr_ref = config['Observations']['SNR']
    z = config['Observations']['z']
    metric = {}
    # processing. Band after band

    Ra_ref = 0.000
    Dec_ref = -2.308039
    time_ref = time.time()

    sql_i = sqlconstraint
    #sql_i += ' AND abs(fieldRA-(%f))< %f' % (Ra_ref, 1.e-2)+' AND '
    #sql_i += 'abs(fieldDec-(%f))< %f' % (Dec_ref, 1.e-2)
    #sql_i += ' AND fieldRA > %f AND fieldRA< %f' % (0., 3.)
    #sql_i += ' AND fieldDec > %f AND fieldDec< %f' % (-10., -5.)
    for band in bands:
        lim_sn[band] = ReferenceData(
            config['Li file'], config['Mag_to_flux file'], band, z)

    metric = module.SNObsRateMetric(lim_sn=lim_sn, names_ref=config['names_ref'], season=season, coadd=config['Observations']
                                    ['coadd'], z=z, bands=bands, snr_ref=dict(zip(bands, snr_ref)))
    bundles.append(metricBundles.MetricBundle(metric, slicer, sql_i))
    names.append(band)

    bdict = dict(zip('snrrate', bundles))

    resultsDb = db.ResultsDb(outDir='None')
    mbg = metricBundles.MetricBundleGroup(bdict, opsimdb,
                                          outDir=outDir, resultsDb=resultsDb)

    mbg.runAll()

    print('processing done')
    # Let us display the results

    for band, val in bdict.items():
        metValues = val.metricValues[~val.metricValues.mask]
        res = None
        # print(metValues)
        for vals in metValues:
            if vals is not None:
                if res is None:
                    res = vals
                else:
                    #print(len(res), len(vals))
                    # print(type(vals))
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
