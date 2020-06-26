import matplotlib.pyplot as plt
from sn_plotter_fitlc.fitlcPlot import FitPlots

dictfiles = {}
x1 = -2.0
color = 0.2

for bluecutoff in [360., 380.0]:
    for ebvofMW in [0.0]:
        thedir = 'Output_Fit_{}_800.0_ebvofMW_{}'.format(bluecutoff, ebvofMW)
        fi_cosmo_cosmo = 'Fit_sn_cosmo_Fake_Fake_DESC_seas_-1_{}_{}_{}_800.0_ebvofMW_{}_sn_cosmo.hdf5'.format(x1, color,
                                                                                                              bluecutoff, ebvofMW)
        fi_fast_cosmo = 'Fit_sn_fast_Fake_Fake_DESC_seas_-1_{}_{}_{}_800.0_ebvofMW_{}_sn_cosmo.hdf5'.format(x1, color,
                                                                                                            bluecutoff, ebvofMW)
        fi_fast_fast = 'Fit_sn_fast_Fake_Fake_DESC_seas_-1_{}_{}_{}_800.0_ebvofMW_{}_sn_fast.hdf5'.format(x1, color,
                                                                                                          bluecutoff, ebvofMW)

        dictfiles['cosmo_cosmo_{}_{}'.format(bluecutoff,
                                             ebvofMW)] = '{}/{}'.format(thedir, fi_cosmo_cosmo)
        """
        dictfiles['fast_cosmo_{}_{}'.format(bluecutoff,
                                            ebvofMW)]='{}/{}'.format(thedir, fi_fast_cosmo)
        dictfiles['fast_fast_{}_{}'.format(bluecutoff,
                                           ebvofMW)]='{}/{}'.format(thedir, fi_fast_fast)
        """

fitplot = FitPlots(dictfiles)
fitplot.plot2D(fitplot.SN_table, 'z', 'Cov_colorcolor',
               '$z$', '$\sigma_{color}$')

plt.show()
