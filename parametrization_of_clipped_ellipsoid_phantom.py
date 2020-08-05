# This script calculates the b parameter of the ellipsoid formula for the clipped ellipsoid phantom model.

import numpy as np
import pandas as pd
import math

# region Importing data and initalizing lists

data = pd.read_csv('C:/Users/Aaron/Desktop/Nanobiology/Year 3/Bachelor End Project/Scripts and Results/Alpha Data, Graphs and Results/First Max Last Major Minor Axis Data/First_Max_Last_Major_Minor_Axis.txt', sep="\t", header=None)
data.columns = ["First Major", "Max Major", "Last Major", "First Minor", "Max Minor", "Last Minor", "Height"]

a = list(data['Max Major'])
x = list(data['First Major'])
y = list(data['Height'])

b = []

z_height = 0.998

for i in range(len(a)):
    a[i] = np.round(a[i], 3)
    x[i] = np.round(x[i], 3)
    y[i] = z_height*(y[i]/2)

# endregion

# region Solving ellipsoid formula for b parameter

for i in range(len(a)):
    dummy = np.power(a[i], 2) - np.power(x[i], 2)
    b_temp = (a[i]*y[i])/(math.sqrt(dummy))
    b.append(b_temp)

b_parameter = np.mean(b)

print(np.round(b_parameter, 3))

#endregion
