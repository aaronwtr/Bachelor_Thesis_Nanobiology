# This code transforms .csv data of cross-sectional damage foci areas into a boxplot.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

# region Typesetting of plots

rc('text', usetex=True)

# endregion

# region Importing data

alpha_sim_data = pd.read_csv('.csv file containg cross-sectional areas', delimiter=';')
alpha_exp_data = pd.read_csv('.csv file containg cross-sectional areas')

alpha_sim_data = alpha_sim_data.loc[:, ~alpha_sim_data.columns.str.contains('^Unnamed')]
alpha_exp_data = alpha_exp_data.loc[:, ~alpha_exp_data.columns.str.contains('^Unnamed')]

# endregion

# region Sampling simulated dataset and processing experimental data

sample_times = 1

sampled_data_table = []

all_data = pd.DataFrame()

for i in range(sample_times):

    random_sample = np.random.randint(1, len(alpha_sim_data), 100)

    sampled_data = np.zeros(len(alpha_sim_data), float)

    for j in range(len(random_sample)):
        sampled_data[j] = list(alpha_sim_data['5000 simulated events'])[random_sample[j]]

    sampled_data = [np.nan if x == 0 else x for x in sampled_data]

    sampled_data_process = list(filter(lambda v: v==v, sampled_data))

    sampled_data_table.append(sampled_data_process)

    all_data['Sampled simulation data (n = 100)'] = sampled_data

sampled_data_table = np.asarray(sampled_data_table)

all_data['Experiment 0.5 Gy data (n = 233)'] = alpha_exp_data

# endregion

# region Saving data and outputting plot

np.savetxt('sampled_data_table.csv', sampled_data_table, delimiter=',')

all_data.boxplot()

plt.grid(None)
plt.ylabel('Cross-sectional area ($\mu$m$^2$)')
#plt.savefig('Figure_6_xray_foci_boxplot.pdf')
plt.show()

# endregion
