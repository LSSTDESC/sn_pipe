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
from sn_maf.sn_tools.sn_cadence_tools import Reference_Data
import healpy as hp
import numpy.lib.recfunctions as rf
import sn_maf.sn_plotters.sn_snrPlotters as sn_plot

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

        lim_sn[band] = Reference_Data(
            config['Li file'], config['Mag_to_flux file'], band, z)

        metric[band] = module.SNSNRMetric(config=config, coadd=config['Observations']
                                          ['coadd'], lim_sn=lim_sn[band], names_ref=config['names_ref'], z=z)
        bundles.append(metricBundles.MetricBundle(metric[band], slicer, sql_i))
        names.append(band)

    bdict = dict(zip(names, bundles))

    resultsDb = db.ResultsDb(outDir='None')
    mbg = metricBundles.MetricBundleGroup(bdict, opsimdb,
                                          outDir=outDir, resultsDb=resultsDb)

    # result = mbg.runAll()
    mbg.runAll()
    # Plot the results:
    # SNR vs MJD for a SN with T0=MJD-10
    # Fake observations corresponding to a "perfect" cadence
    # can be superimposed

    # concatenate the results estimated per band
    metricValues = {}
    data_str = ['snr_obs', 'snr_fakes', 'detec_frac']
    # data_str = ['detec_frac']
    for dstr in data_str:
        metricValues[dstr] = None

    print('processed', time.time()-time_ref)
    # print('ici', bdict.keys())
    for band, val in bdict.items():
        print(band, type(val.metricValues))
        data = val.metricValues[~val.metricValues.mask]
        # print(band, len(data), data[0])
        res = {}
        for dstr in data_str:
            res[dstr] = None
        for val in data:
            for dstr in data_str:
                if res[dstr] is None:
                    res[dstr] = val[dstr]
                else:
                    res[dstr] = np.concatenate((res[dstr], val[dstr]))

        for dstr in data_str:
            res[dstr] = np.unique(res[dstr])
            if metricValues[dstr] is None:
                metricValues[dstr] = res[dstr]
            else:
                metricValues[dstr] = np.concatenate(
                    (metricValues[dstr], res[dstr]))

    snr_obs = metricValues['snr_obs']
    snr_fakes = metricValues['snr_fakes']
    detec_frac = metricValues['detec_frac']

    for inum, (Ra, Dec, season) in enumerate(np.unique(snr_obs[['fieldRA', 'fieldDec', 'season']])):
        idx = (snr_obs['fieldRA'] == Ra) & (
            snr_obs['fieldDec'] == Dec) & (snr_obs['season'] == season)
        sel_obs = snr_obs[idx]
        idxb = (np.abs(snr_fakes['fieldRA'] - Ra) < 1.e-5) & (np.abs(
            snr_fakes['fieldDec'] - Dec) < 1.e-5) & (snr_fakes['season'] == season)
        sel_fakes = snr_fakes[idxb]
        sn_plot.SNRPlot(Ra, Dec, season, sel_obs, sel_fakes, config, metric, z)
        if inum >= 10:
            break

    # print(detec_frac.dtype)

    sn_plot.DetecFracPlot(detec_frac, config['Pixelisation']
                          ['nside'], config['names_ref'])

    # frac_obs = Fraction_Observation(res, config, metric)
    # print(frac_obs)
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
