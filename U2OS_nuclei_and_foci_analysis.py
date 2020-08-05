# This code processes and analyzes U2OS nuclei and damage foci. It can output several characteristics of these
# structures such as area, volume and perimeter.

import cv2
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc


# region Functions

def custom_watershed(img):
    distance = ndimage.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, indices=False, min_distance=400, labels=img)
    markers = ndimage.label(local_maxi)[0]
    return watershed(-distance, markers, mask=img)


def is_gray_scale(img):
    if len(img.shape) < 3:
        return True

    elif len(img.shape) == 3:
        return False

    else:
        return print("Image is not gray scale and not RGB")


def pd_centered(df):
    return df.style.set_table_styles([{"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]}])


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def micrometer_area(px):
    return (16716*px)/(1048576)


def micrometer_perimeter(px):
    return px/(7.92)


def ellipsoid_volume(semiA, semiB, semiC):
    return (4/3)*np.pi*semiA*semiB*semiC


# endregion


# region Number of images to be processed and resolution

num_files = 9
res = 62.73

# endregion

# region Typesetting of plots

rc('text', usetex=True)

#endregion

# region Initalizing some constants and iterables

zstack = 0
temp_df = []

# endregion

# region Analyzing images and obtaining cell or foci statistics

for i in range(num_files):
    start = 1
    print('0.5Gy ' + str(i + 1)+'.bmp')
    path4 = '.bmp image path ' + str(i + start) + '.bmp'
    zstack = zstack + 1

    image = cv2.imread(path4)

    all_semisA = []
    all_semisB = []

    depth_measured = 0.988
    depth = ((num_files)/2)*0.988

    cv2.imshow("Pre bin and thresh", image)
    cv2.waitKey(1)

    cv2.imwrite("pre_bin_and_thresh.png", image)

    if type(image) is not None:

        if not is_gray_scale(image):
            binary_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        else:
            binary_image = image

        gray = cv2.GaussianBlur(binary_image, (13, 13), 0)
        median = cv2.medianBlur(gray, 15)

        mitotic_cells_gray = gray

        retval, binary_image_temp = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        skernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image_temp, cv2.MORPH_CLOSE, skernel, iterations=1)

        cv2.imshow("Binary", binary_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        cv2.imwrite("binarized_and_tresholded.png", binary_image)

        cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Binary', 600, 600)

        edged = cv2.Canny(binary_image, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cv2.imshow("Contourization", edged)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        cv2.imwrite("contourized.png", edged)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        pixel_per_metric = None

        cell_name = []
        cell_area = []
        cell_perimeter = []

        count = 0

        semisA = []
        semisB = []

        major = []
        minor = []

        volume = []

        cx = []
        cy = []

        zstack_list = []

        for c in cnts:
            if cv2.contourArea(c) < 1000:      # Change value to +/- 1000 for Nucleus. +/- 50 for damage foci
                continue

            count = count + 1

            cell_name.append("Cell " + str(count))
            cell_area.append(np.round(cv2.contourArea(c), 2)/res)
            volume.append((cv2.contourArea(c)/res)*depth_measured)
            cell_perimeter.append(cv2.arcLength(c, True))

            orig = image.copy()  # Defining a boxed coordinate system around cell for calculations
            box = cv2.minAreaRect(c)
            box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)

            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            cnt = c
            M = cv2.moments(cnt)

            cx.append(int(M['m10'] / M['m00']))
            cy.append(int(M['m01'] / M['m00']))

            for (x, y) in box:  # Drawing origin points
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box  # Defining top left, top right, bottom right and bottom left points
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dimA = micrometer_perimeter(dist.euclidean((tltrX, tltrY), (blbrX, blbrY)))
            dimB = micrometer_perimeter(dist.euclidean((tlblX, tlblY), (trbrX, trbrY)))

            semisA.append(dimA/2)
            semisB.append(dimB/2)

            zstack_list.append(zstack)

            if tlblY <= 84:
                cv2.putText(orig, "{:.1f}um".format(dimA), (int(tltrX - 15), int(tltrY + 175)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2)

            else:
                cv2.putText(orig, "{:.1f}um".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2)

            cv2.putText(orig, "{:.1f}um".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)

            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image', 600, 600)

        all_data_cell_semis = pd.DataFrame()

        for axis in range(len(semisA)):
            if semisA[axis] > semisB[axis]:
                major.append(semisA[axis])
                minor.append(semisB[axis])
            else:
                major.append(semisB[axis])
                minor.append(semisA[axis])

        all_data_cell_semis['Major axis'] = major
        all_data_cell_semis['Minor axis'] = minor
        all_data_cell_semis['CX'] = cx
        all_data_cell_semis['CY'] = cy
        all_data_cell_semis['Area'] = cell_area
        all_data_cell_semis['Volume'] = volume
        all_data_cell_semis['Z-Stack'] = zstack_list

        temp_df.append(all_data_cell_semis)

# endregion

# region Reshaping data for clean dataframe output sorted by cell or z-stack

temp_df_x = pd.concat(temp_df)
temp_df_x = temp_df_x.reset_index(drop=True)

binwidth = 28

temp_df_x = temp_df_x.groupby('CX')

main_df_x = pd.DataFrame()

for key, item in temp_df_x:
    main_df_x = main_df_x.append(temp_df_x.get_group(key))

temp_df = main_df_x

temp_df = temp_df.groupby('CY')

main_df = pd.DataFrame()

for key, item in temp_df:
    main_df = main_df.append(temp_df.get_group(key))

cell_number = 1

cell_number_list = []

for i in range(len(main_df['CX']) - 1):
    if list(main_df['CX'])[i + 1] - list(main_df['CX'])[i] < 10 and list(main_df['CY'])[i + 1] - list(main_df['CY'])[i] < 10:
        cell_number_list.append(cell_number)
    else:
        cell_number_list.append(cell_number)
        cell_number = cell_number + 1

cell_number_list.append(cell_number)

main_df["Cell"] = cell_number_list

main_df = main_df.reset_index(drop=True)

cell_number_list_no_dup = list(dict.fromkeys(cell_number_list))

cell_indices = []

for i in range(len(cell_number_list_no_dup)):
    cell_indices.append(main_df.index[main_df["Cell"] == cell_number_list_no_dup[i]])

total_volume = []

for i in range(len(cell_indices)):
    total_volume.append(sum(list(main_df["Volume"])[cell_indices[i][0]:cell_indices[i][-1] + 1]))

max_major = []
max_minor = []
max_area = []

count = 0

for i in range(len(cell_indices)):
    temp_check = list(main_df["Area"])[cell_indices[i][0]:cell_indices[i][-1] + 1]

    for j in range(len(temp_check)):
        if temp_check[j] == max(list(main_df["Area"])[cell_indices[i][0]:cell_indices[i][-1] + 1]):
            max_major.append(list(main_df["Major axis"])[count])
            max_minor.append(list(main_df["Minor axis"])[count])
            max_area.append(list(main_df["Area"])[count])

        count = count + 1

fitted_volume = []

for i in range(len(max_major)):
    fitted_volume.append(ellipsoid_volume(max_major[i], max_minor[i], depth/2))

major_axes = []
minor_axes = []

plotted_cell = 9

for i in range(len(main_df)):
    if list(main_df['Cell'])[i] == plotted_cell:
        major_axes.append(list(main_df['Major axis'])[i])
        minor_axes.append(list(main_df['Minor axis'])[i])


cell_slices = np.arange(len(major_axes))

main_df_cell_sort = main_df

main_df_zstack_sort = main_df.sort_values(by=['Z-Stack'])

print(main_df_zstack_sort)

print(main_df_cell_sort)

np.savetxt('U2Os_Analysis_Sorted_By_Cell.csv', main_df_cell_sort.values, fmt='%1.3f\t%1.3f\t%d\t%d\t%1.3f\t%1.3f\t%d\t%d', delimiter="\t")

np.savetxt('U2Os_Analysis_Sorted_By_ZStack.txt', main_df_zstack_sort.values, fmt='%1.3f\t%1.3f\t%d\t%d\t%1.3f\t%1.3f\t%d\t%d', delimiter="\t")

# endregion

# region Plotting major- and minor axis distributions to assess cell phantom shape

plt.scatter(cell_slices, major_axes, label='Major axis', color='#808080')
plt.scatter(cell_slices, minor_axes, label='Minor axis', color='#D3D3D3')
plt.plot(cell_slices, major_axes, '--', color='#808080')
plt.plot(cell_slices, minor_axes, '--', color='#D3D3D3')
plt.xlabel('Z-Slice')
plt.ylabel('Axis length ($\mu$m)')
plt.legend()
plt.title('Distribution of the major and minor axes')
plt.show()
plt.clf()

# endregion

# region Reshaping data to display maximum major and minor axis, mean major and minor axis and minimum major and minor axis

mean_major_axis = np.mean(list(main_df['Major axis'].to_list()))
mean_minor_axis = np.mean(list(main_df['Minor axis'].to_list()))

cell_separate_major = []
cell_separate_minor = []
cell_separate_zstack = []

for i in range(len(cell_number_list_no_dup)):
    cell_separate_major_temp = []
    cell_separate_minor_temp = []
    cell_separate_zstack_temp = []
    for j in range(len(main_df_zstack_sort)):
        if cell_number_list_no_dup[i] == list(main_df_zstack_sort["Cell"])[j]:
            cell_separate_major_temp.append(list(main_df_zstack_sort["Major axis"])[j])
            cell_separate_minor_temp.append(list(main_df_zstack_sort["Minor axis"])[j])
            cell_separate_zstack_temp.append(list(main_df_zstack_sort["Z-Stack"])[j])

    cell_separate_major.append(cell_separate_major_temp)
    cell_separate_minor.append(cell_separate_minor_temp)
    cell_separate_zstack.append(cell_separate_zstack_temp)

height = []

for i in range(len(cell_separate_zstack)):
    if len(cell_separate_major[i]) > 3:
        height.append(cell_separate_zstack[i][-1] - cell_separate_zstack[i][0])

first_max_last_cell_major = []
first_max_last_cell_minor = []

for i in range(len(cell_separate_major)):
    first_max_last_cell_major_temp = []
    first_max_last_cell_minor_temp = []
    if len(cell_separate_major[i]) > 3:
        first_max_last_cell_major_temp.append(cell_separate_major[i][0])
        first_max_last_cell_major_temp.append(max(cell_separate_major[i]))
        first_max_last_cell_major_temp.append(cell_separate_major[i][-1])
        first_max_last_cell_major.append(first_max_last_cell_major_temp)

        first_max_last_cell_minor_temp.append(cell_separate_minor[i][0])
        first_max_last_cell_minor_temp.append(max(cell_separate_minor[i]))
        first_max_last_cell_minor_temp.append(cell_separate_minor[i][-1])
        first_max_last_cell_minor.append(first_max_last_cell_minor_temp)


first_max_last_df = pd.DataFrame()      # COMMENT IF ANALYZING FOCI
max_foci_df = pd.DataFrame()            # COMMENT IF ANALYZING NUCLEI

first_list = []
max_list = []
last_list = []

for i in range(len(first_max_last_cell_major)):
    first_list.append(first_max_last_cell_major[i][0])
    max_list.append(first_max_last_cell_major[i][1])
    last_list.append(first_max_last_cell_major[i][2])

first_max_last_df['First Major Axis'] = first_list
first_max_last_df['Max Major Axis'] = max_list
first_max_last_df['Last Major Axis'] = last_list

first_list = []
max_list = []
last_list = []

for i in range(len(first_max_last_cell_minor)):
    first_list.append(first_max_last_cell_minor[i][0])
    max_list.append(first_max_last_cell_minor[i][1])
    last_list.append(first_max_last_cell_minor[i][2])

first_max_last_df['First Minor Axis'] = first_list
first_max_last_df['Max Minor Axis'] = max_list
first_max_last_df['Last Minor Axis'] = last_list
first_max_last_df['Height'] = height

max_foci_df['Max Damage Foci'] = max_list

np.savetxt('cross_sectional_surface_area_damage_foci.txt', max_area)

# endregion
