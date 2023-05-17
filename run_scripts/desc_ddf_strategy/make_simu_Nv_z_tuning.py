import pandas as pd


names = ['tagName', 'plotName', 'Nvisits_u', 'Nvisits_g', 'Nvisits_r', 'Nvisits_i', 'Nvisits_z', 'Nvisits_y',
         'cadence_g', 'cadence_r', 'cadence_i', 'cadence_z', 'cadence_y', 'moonPhaseu', 'moonswapFilter']

# get ref numbers
Nu = 7
Ng = 4
Nr = 18
Ni = 90
Nz = 128
Ny = 38

cadg = 2
cadr = 2
cadi = 2
cadz = 2
cady = 2

moonPhaseu = 20.
moonswapFilter = 'y'

r = []
io = 0
for NNi in range(Ni, int(1.1*Ni), 1):
    for NNz in range(Nz, int(1.1*Nz), 1):
        for NNy in range(Ny, int(1.1*Ny), 1):
            io += 1
            combi_name = 'combi_{}'.format(io)
            plot_name = 'plot_{}'.format(io)
            r.append((combi_name, plot_name, Nu, Ng, Nr, NNi, NNz, NNy,
                     cadg, cadr, cadi, cadz, cady, moonPhaseu, moonswapFilter))

df = pd.DataFrame(r, columns=names)

print(df)

df.to_csv('combi1.csv', index=False)
