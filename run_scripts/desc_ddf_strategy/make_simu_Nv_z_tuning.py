import pandas as pd
from optparse import OptionParser

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
if zcomp == '0.80':
    Ni = 90
    Nz = 128
    Ny = 38

if zcomp == '0.75':
    Ni = 88
    Nz = 76
    Ny = 30

if zcomp == '0.70':
    Ni = 66
    Nz = 62
    Ny = 12

cadg = 2
cadr = 2
cadi = 2
cadz = 2
cady = 2

moonPhaseu = -1.0
moonswapFilter = 'y'

r = []
io = 0
for NNi in range(int(0.9*Ni), int(1.2*Ni), 2):
    for NNz in range(int(0.9*Nz), int(1.2*Nz), 2):
        for NNy in range(int(0.9*Ny), int(1.2*Ny), 2):
            io += 1
            combi_name = 'combi_{}'.format(io)
            plot_name = 'plot_{}'.format(io)
            r.append((combi_name, plot_name, Nu, Ng, Nr, NNi, NNz, NNy,
                     cadg, cadr, cadi, cadz, cady, moonPhaseu, moonswapFilter))

df = pd.DataFrame(r, columns=names)

print(df)

df.to_csv('combi_{}.csv'.format(zcomp), index=False)
