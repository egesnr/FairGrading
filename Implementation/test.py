import csv
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd


# given 1D array checks intersection of both sources
# empty values must be nan
def intersection_of_grades(src_a, src_b, n):
    for i in range(n):
        np.isnan(src_a)

    return


# rows = []
# with open("Grading_Assignment.csv", "r") as file:
#     read = csv.reader(file)
#
#     for row in read:
#         rows.append(row)
#
# nump_arr = np.array(rows)
# x_value = nump_arr[0:1,1:]
# y_value = nump_arr[:, 1:]
#
# sc1 = nump_arr[1:,1:2]
#
# df = pd.read_csv("Grading_Assignment.csv")
# instructors_grades_notEmpty = list()
"""
for i in range(1,25):
    temp = df["sc"+str(i)].to_numpy()
    instructors_grades_notEmpty = np.append()
    """
# instructor2 = df["sc2"].to_numpy()
# instructor1 = df["sc1"].to_numpy()
# instructor3 = df["sc3"].to_numpy()
# instructor4 = df["sc4"].to_numpy()
# instructor5 = df["sc5"].to_numpy()

# isnan function is to check whatever the content of the array is empty
# Then because of list points empty values, logical_not func is used
# instructor1_notEmp = instructor1[np.logical_not(np.isnan(instructor1))]
# instructor2_notEmp = instructor2[np.logical_not(np.isnan(instructor2))]
#
# plot.title("Instructor's grades")
# plot.ylabel("Grades")
# plot.xlabel("Instructors")
# plot.scatter(np.ones(instructor1_notEmp.size),instructor1_notEmp)
# plot.show()

df = pd.read_csv("Grading_Assignment.csv")
df2 = df.drop('ID', inplace=False, axis=1)

# combination = np.array([[]])
# combinations = np.array(range(1, 1001), ndmin=2)
# combinations = combinations.T
# for i in range(24):
#     for j in range(i + 1, 25):
#         combination = np.array(df2.iloc[:, [i, j]])
#
#         combinations = np.column_stack((combinations, combination))

for i in range(24):
    for j in range(i + 1, 25):
        combinations = df2.iloc[:, [i, j]]
        commons = combinations.dropna()
        print(commons.iloc[0].mean(), commons.iloc[1].mean())
        
