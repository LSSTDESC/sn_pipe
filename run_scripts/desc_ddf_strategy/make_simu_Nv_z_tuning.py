import pandas as pd
from optparse import OptionParser
import numpy as np

parser = OptionParser(description='Script to generate fake obs, simu and fit')

parser.add_option('--zcomp', type=str, default='0.80',
                  help='rdshift completeness [%default]')
opts, args = parser.parse_args()

names = ['tagName', 'plotName', 'Nvisits_u', 'Nvisits_g',
         'Nvisits_r', 'Nvisits_i', 'Nvisits_z', 'Nvisits_y',
         'cadence_g', 'cadence_r', 'cadence_i', 'cadence_z',
         'cadence_y', 'moonPhaseu', 'moonswapFilter']

zcomp = opts.zcomp

# get ref numbers
Nu = 7
Ng = 4
Nr = 18
pmin = 0.9
pmax = 1.2
if zcomp == '0.80':
    Ni = 90
    Nz = 128
    Ny = 38
    pmax = 1.1

if zcomp == '0.75':
    Ni = 88
    Nz = 76
    Ny = 30

if zcomp == '0.70':
    Ni = 66
    Nz = 62
    Ny = 12

if zcomp == '0.65':
    Ng = 4
    Nr = 9
    Ni = 10
    Nz = 15
    Ny = 3

if zcomp == '0.60':
    Ng = 4
    Nr = 9
    Ni = 1
    Nz = 1
    Ny = 0

if zcomp == '0.55':
    Ng = 4
    Nr = 7
    Ni = 1
    Nz = 1
    Ny = 0

cadg = 2
cadr = 2
cadi = 2
cadz = 2
cady = 2

moonPhaseu = -1.0
moonswapFilter = 'y'

r = []
io = 0
Ni_min = np.max([int(pmin*Ni), Ni-3])
Ni_max = np.max([int(pmax*Ni), Ni+3])
Nz_min = np.max([int(pmin*Nz), Nz-3])
Nz_max = np.max([int(pmax*Nz), Nz+3])
Ny_min = np.max([int(pmin*Ny), Ny-3])
Ny_max = np.max([int(pmax*Ny), Ny+3])


for NNi in range(Ni_min, Ni_max, 1):
    for NNz in range(Nz_min, Nz_max, 1):
        for NNy in range(Ny_min, Ny_max, 1):
            io += 1
            combi_name = 'combi_{}'.format(io)
            plot_name = 'plot_{}'.format(io)
            r.append((combi_name, plot_name, Nu, Ng, Nr, NNi, NNz, NNy,
                     cadg, cadr, cadi, cadz, cady, moonPhaseu, moonswapFilter))

df = pd.DataFrame(r, columns=names)

print(df)

df.to_csv('combi_{}.csv'.format(zcomp), index=False)
