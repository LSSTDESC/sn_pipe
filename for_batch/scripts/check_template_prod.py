import glob
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color[%default]")
parser.add_option("--zmax", type=float, default=1.01,
                  help="SN max redshift[%default]")
parser.add_option("--zmin", type=float, default=0.01,
                  help="SN min redshift[%default]")
parser.add_option("--zstep", type=float, default=0.01,
                  help="SN min redshift[%default]")
parser.add_option("--error_model", type=int, default=1,
                  help="error model for SN LC estimation[%default]")
parser.add_option("--bluecutoff", type=float, default=380.0,
                  help="blue cutoff[%default]")
parser.add_option("--redcutoff", type=float, default=800.0,
                  help="red cutoff[%default]")

opts, args = parser.parse_args()

x1 = opts.x1
color = opts.color

zmin = np.round(opts.zmin,2)
zmax = np.round(opts.zmax,2)
zstep = np.round(opts.zstep,2)


nvals = int((zmax-zmin)/zstep)+1

ebvs = np.arange(0.0,0.30,0.01)

cutoff = '{}_{}'.format(opts.bluecutoff, opts.redcutoff)
if opts.error_model>0:
    cutoff = 'error_model'

prefix = '/sps/lsst/users/gris/fakes_for_templates_{}_ebvofMW_'.format(cutoff)

listref = list(np.arange(opts.zmin,opts.zmax,opts.zstep))
listref = np.round(listref,2)

for ebv in ebvs:
    ebv = np.round(ebv,2)
    fis = glob.glob('{}{}/fake_simu_data/{}_{}/LC_Fake_{}_{}*.hdf5'.format(prefix,ebv,x1,color,x1,color))
    if len(fis) != nvals:
        print('problem here',ebv,len(fis),nvals)
        rh = []
        for ff in fis:
            z = ff.split('/')[-1]
            z = z.split('_')[4]
            #print(z)
            rh.append(np.round(float(z),2))
        print(set(listref).difference(set(rh)))
