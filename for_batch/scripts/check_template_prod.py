import glob
import numpy as np


x1 = 0.0
color = 0.0

nvals = 119
ebvs = np.arange(0.0,0.40,0.01)
prefix = '/sps/lsst/users/gris/fakes_for_templates_380.0_800.0_ebvofMW_'

listref = list(np.arange(0.0,1.2,0.01))
listref[0] = 0.01
listref = np.round(listref,2)
for ebv in ebvs:
    ebv = np.round(ebv,2)
    fis = glob.glob('{}{}/fake_simu_data/{}_{}/LC_Fake_{}_{}*.hdf5'.format(prefix,ebv,x1,color,x1,color))
    if len(fis) != nvals:
        print('problem here',ebv,len(fis))
        rh = []
        for ff in fis:
            z = ff.split('/')[-1]
            z = z.split('_')[4]
            #print(z)
            rh.append(np.round(float(z),2))
        print(set(listref).difference(set(rh)))
