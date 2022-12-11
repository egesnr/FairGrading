import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
rows = []
with open("Implementation\Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)

'''
df = pd.read_csv("Grading_Assignment.csv")


def infos():
    # First 10 row of the dataset
    print(df.head(10))
    print(df.shape)
    print(df.describe())
    # Column of the first teacher
    teacher1 = df['sc1']
    # Column of the second one
    teacher2 = df['sc2']

    # Mean value of teacher 1
    print("mean of teacher 1 = ", df["sc1"].mean())


# Dropped ID Column for d2 dataframe
df2 = df.drop('ID', inplace=False, axis=1)


def calculate_mean():
    # Calculate mean value of every teacher
    means = []
    for i in df2:
        print("mean of ", i, " is ", df2[i].mean())
        means.append(df2[i].mean())
    return means


def plotting():
    # Comparing values of sc1 and sc2 by histogram
    plt.figure("2")
    plt.title('Grading Dataset')
    plt.ylabel('Frequency')
    plt.xlabel('Grades')
    #   plt.hist(teacher1)
    # plt.hist(teacher2,color = 'green',alpha = 0.5)
    # plt.show()

    plt.figure("1")
    plt.title("Instructor's grades")
    plt.ylabel("Means")
    plt.xlabel("Instructors")
    x = np.arange(1, 26)
    plt.scatter(x, calculate_mean)
    plt.xticks(np.arange(1, 26))
    # plt.show()


'''
for i in range(24):
    for j in range(i+1,25):
      combinations = df2.iloc[:, [i, j]]
      commons = combinations.dropna()
      teacher_x = commons.iloc[0].mean()
      teacher_y = commons.iloc[1].mean()
'''
a = []
deviation = []
for i in range(25):
    for j in range(25):
        combinations = df2.iloc[:, [i, j]]
        commons = combinations.dropna()
        teacher_x = commons.iloc[:, 0].mean()
        teacher_y = commons.iloc[:, 1].mean()
        mean = teacher_x - teacher_y
        a.append(mean)
        diff = (np.array(commons.iloc[:, 0]) - np.array(commons.iloc[:, 1]))
        deviation.append(np.sqrt(np.sum((diff - mean) ** 2) / 25.0))

deviation = np.array(deviation).reshape(25, 25)
b = np.array(a)
c = b.reshape(25, 25)

d = pd.DataFrame(c)
deviation = pd.DataFrame(deviation)
#print(d)
print(deviation)