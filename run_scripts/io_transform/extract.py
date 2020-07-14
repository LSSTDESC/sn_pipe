import numpy as np
from optparse import OptionParser
from sn_tools.sn_io import getObservations
from sn_tools.sn_obs import renameFields, patchObs
import numpy.lib.recfunctions as rf

parser = OptionParser()

parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',
                  help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='db',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str",
                  default='../../DB_Files', help="db dir [%default]")
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - for WDF only[%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - for WDF only[%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - for WDF only[%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - for WDF only[%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type DD or WFD[%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

dbName = opts.dbName
dbExtens = opts.dbExtens
dbDir = opts.dbDir
RAmin = opts.RAmin
RAmax = opts.RAmax
Decmin = opts.Decmin
Decmax = opts.Decmax
fieldType = opts.fieldType
nside = 64

# loading observations

observations = getObservations(dbDir, dbName, dbExtens)
orig_names = observations.dtype.names

print('before renaming', observations.dtype)
# rename fields

observations = renameFields(observations)

RACol = 'fieldRA'
DecCol = 'fieldDec'

if 'RA' in observations.dtype.names:
    RACol = 'RA'
    DecCol = 'Dec'

observations, patches = patchObs(observations, fieldType,
                                 dbName,
                                 nside,
                                 RAmin, RAmax,
                                 Decmin, Decmax,
                                 RACol, DecCol,
                                 display=False)

# drop fields not originally present
observations = rf.drop_fields(
    observations, ['healpixID', 'pixRA', 'pixDec', 'ebv'])

# rename fields that were modified in the process

observations = rf.rename_fields(
    observations, {'fieldRA': 'RA',
                   'fieldDec': 'Dec',
                   'visitExposureTime': 'exptime',
                   'observationStartMJD': 'mjd',
                   'filter': 'band'})

# check that erverything is fine
print(set(observations.dtype.names)-set(orig_names))
assert(set(observations.dtype.names) == set(orig_names))
print(len(observations), observations.dtype)
"""
import matplotlib.pyplot as plt
plt.plot(observations['fieldRA'], observations['fieldDec'], 'ko')

plt.show()
"""

np.save('{}_{}.npy'.format(dbName, fieldType), np.copy(observations))
