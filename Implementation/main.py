import numpy as np
import pandas as pd
import csv
import difflib

import pandas as pd
def csv_read(csv_file):
    rows = []
    with open(str(csv_file), "r") as file:
        csv_read = csv.reader(file)
        for row in csv_read:
            rows.append(row)
    return rows

#print(csv_read("Grading Assignment.csv"))


csv_file = csv_read("Grading Assignment.csv")
df = pd.read_csv("Grading Assignment.csv")
df2 = df.drop('ID', inplace=False, axis=1)

df2_transposed = df2.transpose()
print(df2_transposed)


def Find_Grades():
    common_grades = []
    for x in range (len(df2_transposed.columns)):
        decoy_cammon = []
        for y in range (len(df2_transposed)):
            cell = df2_transposed.iloc[y , x]
            if pd.notna(cell):
                decoy_cammon.append(cell)
        common_grades.append(decoy_cammon)
    return common_grades


def Find_Instructors():
    instructors = []
    for x in range(len(df2_transposed.columns)):
        decoy = []
        for y in range(len(df2_transposed)):
            cell = df2_transposed.iloc[y, x]
            if pd.notna(cell):
                decoy.append(y+1)
        instructors.append(decoy)
    return instructors

all_gardes = Find_Grades()
all_instructors = Find_Instructors()

a = pd.DataFrame(all_gardes)
b = pd.DataFrame(all_instructors)
print(b)


def Find_Mean():
    a = pd.DataFrame(all_gardes)
    for i in range(len(a)):
        a['mean'] = a.mean(axis=1)
    return a

'''
print(b)
print(len(b))
print(b.columns)
'''
'''
b_array = b.iloc[0]
b_array2 =b.iloc[1]
sm = difflib.SequenceMatcher(None, b_array , b_array2)
print("************")
print(sm.ratio())
print(sm.get_matching_blocks())
'''

def Cammon_Ratio():
    common_ratio = []
    b = pd.DataFrame(all_instructors)
    for x in range(1):
        instructor1 = b.iloc[x]
        for i in range(len(b)):
            other = b.iloc[i]
            sm = difflib.SequenceMatcher(None, instructor1, other)
            common_ratio.append(sm.ratio())
    return common_ratio

c = Cammon_Ratio()
print(c)


smilarity_array = []
for i in range(len(c)):
     element = c[i]
     if element  == 1.0 :
         continue
     if element > 0.0:
         new_element = element
         element = c[i+1]
         if new_element > element or new_element == element:
            smilarity_array.append(i)







    #print(type(df2))

'''
df2_transposed = df2.transpose()
print(df2_transposed)
for x in range(len(df2_transposed.columns)):
    for y in range(len(df2_transposed)):
        cell = df2_transposed.iloc[y,x]
        #if pd.notna(cell):
        print(cell)
'''

'''
def meanCorrelation(train_data):
    a = []
    train_data = pd.DataFrame(train_data)
    for i in range(len(train_data.columns)):
        for j in range(len(train_data.columns)):
            combinations = train_data.iloc[:, [i, j]]
            commons = combinations.dropna()
            teacher_x = commons.iloc[:, 0].mean()
            teacher_y = commons.iloc[:, 1].mean()
            a.append(teacher_x - teacher_y)

    b = np.array(a)
    c = b.reshape(25, 25)
    d = pd.DataFrame(c)
    return d

print(meanCorrelation(df2))
'''

'''
np_row1 = np.array(row1)
np_row2 = np.array(row2)
#print(np_row1)
#print(np_row2)

commonInstructor_list = np.array([])
for x in range(len(df2)):
    selected_row1 = df2.iloc[x]
    np_row1 = np.array(selected_row1)
    for y in range(len(df2)):
        decoy_list  = np.array([])
        for i in range(len(np_row1)):
            selected_row2 = df2.iloc[y]
            np_row2 = np.array(selected_row2)

            if i  == len(np_row1-1):
                commonInstructor_list.append(decoy_list)
                #a = np.append(commonInstructor_list , decoy_list)

            if str(np_row1[i]) == 'nan' or str(np_row2[i]) == "nan":
                continue

            if str(np_row1[i]) != 'nan' and str(np_row2[i]) != "nan":
                get_index = i+1             #probably instructor
                #decoy_list.append(get_index)
                #b = np.append(decoy_list , get_index)

'''


'''
def calculate_mean():
    # Calculate mean value of every teacher
    means = []
    for i in df2:
        means.append(df2[i].mean())
    return means

'''



