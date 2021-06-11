from sn_design_dd_survey.visuLC import VisuLC
import numpy as np
import pandas as pd


def plot(lcproc, z, band, Nvisits, m5_singleExposure):

    idx = m5_singleExposure['filter'] == band

    lcproc.plotBand(z, band, Nvisits,
                    m5_singleExposure[idx]['fiveSigmaDepth'].tolist()[0])


m5_fName = 'dd_design/m5_files/medValues_flexddf_v1.4_10yrs_DD.npy'
m5_file = pd.DataFrame(np.load(m5_fName, allow_pickle=True))

m5_singleExposure = m5_file.groupby(['filter']).median().reset_index()
print(m5_file.columns)

lcproc = VisuLC(dirTemplates='dd_design/Templates')

# lcproc.plot(0.9)

plot(lcproc, 0.9, 'y', 10, m5_singleExposure)
