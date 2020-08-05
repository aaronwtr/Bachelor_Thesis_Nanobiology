# This script shows the energy deposition output of Geant4-DNA both as a 3D scatter plot and a Y, Z plot.

import matplotlib.pyplot as plt
from matplotlib import rc

# region Typesetting of plots

rc('text', usetex=True)

# endregion

# region Extracting and processing data

path = "Geant4-DNA output data"

f = open(path, "r")
lines = f.readlines()
x_pos = []
y_pos = []
z_pos = []
hist_number = []

for x in lines:
    x_pos.append(float(x.split(' ')[5]))
    y_pos.append(float(x.split(' ')[6]))
    z_pos.append(float(x.split(' ')[7]))
    hist_number.append(x.split(' ')[10])
f.close()

hist_number_stripped = []

for i in range(len(hist_number)):
    hist_number_stripped.append(hist_number[i].strip())

hist_number_stripped_noduplicates = list(dict.fromkeys(hist_number_stripped))

none_list = [None for x in hist_number_stripped_noduplicates]

indices = dict(zip(hist_number_stripped_noduplicates, none_list))

tag = 0

for i in range(len(hist_number_stripped)-1):
    if hist_number_stripped[i] != hist_number_stripped[i+1]:
        indices[hist_number_stripped[i]] = tag
        tag = tag + 1
    else:
        tag = tag + 1

indices[hist_number_stripped[-1]] = len(hist_number_stripped)

# endregion

# region 3D scatter

X = x_pos[0:indices[hist_number_stripped[0]]]
Y = y_pos[0:indices[hist_number_stripped[0]]]
Z = z_pos[0:indices[hist_number_stripped[0]]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter3D(X, Y, Z, color='grey', alpha=0.08)
ax.set_xlabel('X-position ($\mu$m)')
ax.set_ylabel('\n\nY-position ($\mu$m)')
ax.set_zlabel('\n\nZ-position ($\mu$m)')

plt.show()

# endregion

# region Y, Z scatter

ax = plt.subplot(111)
ax.scatter(Y, Z, color='grey', alpha=0.08)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel('\nY-position ($\mu$m)')
plt.ylabel('\nZ-position ($\mu$m)')
plt.grid(True)

plt.show()

# endregion
