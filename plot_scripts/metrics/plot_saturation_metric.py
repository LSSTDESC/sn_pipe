from sn_tools.sn_io import loopStack


fName = 'descddf_v1.5_10yrs_SaturationMetric_DD_nside_128_coadd_1_0.0_360.0_-1.0_-1.0_npixels_3_ebvofMW_0.0_0.hdf5'
fi = 'MetricOutput/descddf_v1.5_10yrs/Saturation_COSMOS/{}'.format(fName)

res = loopStack([fi], 'astropyTable')

print(res)
