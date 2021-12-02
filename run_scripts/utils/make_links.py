import os
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--Ny", type=int, default=20,
                  help="y-band max visits at z=0.9 [%default]")
parser.add_option("--dirFiles", type=str, default='Fakes_nosigmaInt/Fit',
                  help="main dir for files [%default]")

opts, args = parser.parse_args()

Ny = opts.Ny
dirFiles = opts.dirFiles

newDir = '{}_Ny_{}'.format(dirFiles, Ny)
if not os.path.isdir(newDir):
    os.makedirs(newDir)

for vv in np.arange(0.50, 0.95, 0.05):
    tt = str(np.round(vv, 2))
    tt = tt[::-1].zfill(4)[::-1]
    ff = 'DD_{}_Ny_{}'.format(tt, Ny)
    _cmd = 'ln -fs {}/{} {}/DD_{}'.format(dirFiles, ff, newDir, tt)
    print(_cmd)
    os.system(_cmd)

_cmd = 'ln -fs {}/WFD_0.20 {}/.'.format(dirFiles, newDir)
print(_cmd)
os.system(_cmd)
