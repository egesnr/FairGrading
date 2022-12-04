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
df = pd.read_csv("Implementation\Grading_Assignment.csv")
print(df)
#Column of the first teacher
teacher1 = df['sc1']
#Column of the second one
teacher2 = df['sc2']

#Mean value of teacher 1
print("mean of teacher 1 = ",df["sc1"].mean())



#Dropped ID Column for d2 dataframe
df2 = df.drop('ID', inplace=False, axis=1)
print(df2.head())
#Calculate mean value of every teacher 
means = []
for i in df2:
    print("mean of ",i ," is ",df2[i].mean())
    means.append(df2[i].mean())
#Comparing values of sc1 and sc2 by histogram
'''
plt.title('Grading Dataset')
plt.ylabel('Frequency')
plt.xlabel('Grades')
plt.hist(teacher1)
plt.hist(teacher2,color = 'green',alpha = 0.5)
#plt.show()
'''
plt.title("Instructor's grades")
plt.ylabel("Means")
plt.xlabel("Instructors")
x = np.arange(1,26)
plt.scatter(x,means)
plt.xticks(np.arange(1,26))
plt.show()
