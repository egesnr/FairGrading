import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

'''
rows = []
with open("Implementation\Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)

'''
#df = pd.read_csv("Grading_Assignment.csv")
df = pd.read_csv("Implementation\Grading_Assignment.csv")


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
def dev():
 deviation = []
 for i in range(25):
    for j in range(25):
        combinations = df2.iloc[:, [i, j]]
        commons = combinations.dropna()
        teacher_x = commons.iloc[:, 0].mean()
        teacher_y = commons.iloc[:, 1].mean()
        mean = teacher_x - teacher_y
        diff = (np.array(commons.iloc[:, 0]) - np.array(commons.iloc[:, 1]))
        deviation.append(np.sqrt(np.sum((diff - mean) ** 2) / 25.0))

 deviation = np.array(deviation).reshape(25, 25)
 deviation = pd.DataFrame(deviation)
 

def meanCorrelation():
 a = []
 for i in range(25):
    for j in range(25):
      combinations = df2.iloc[:, [i, j]]
      commons = combinations.dropna()
      teacher_x = commons.iloc[:,0].mean()
      teacher_y = commons.iloc[:,1].mean()
      a.append(teacher_x-teacher_y)

 b = np.array(a)
 c = b.reshape(25,25)

 d = pd.DataFrame(c)
 print(d)



def meanCorrelation_train():
 a = []
 for i in range(25):
    for j in range(25):
      combinations = df2.iloc[:, [i, j]]
      commons = combinations.dropna()
      teacher_x = commons.iloc[:-1,0].mean()
      teacher_y = commons.iloc[:-1,1].mean()
      a.append(teacher_x-teacher_y)

 b = np.array(a)
 c = b.reshape(25,25)

 d = pd.DataFrame(c)
 
 return d


def test():
 a = []
 a2 = []
 sum = 0
 sum2 = 0
 
 x_sum = 0
 x_array = []
 for i in range(25):
    
    for j in range(i+1,25):
      combinations = df2.iloc[:, [i, j]]
      commons = combinations.dropna()
      length = len(commons)
      
      random_number1 = random.randint(0,length-1)
      random_number2 = random.randint(0,length-1)
      teacher_x = commons.iloc[random_number1,0]
      teacher_y = commons.iloc[random_number2,1]

      teacher_xx = commons.iloc[:,0]
      teacher_yy = commons.iloc[:,1]
      
      teacher_xx = teacher_xx.loc[teacher_xx!=teacher_x].mean()
      teacher_yy = teacher_yy.loc[teacher_yy!=teacher_y].mean()
     
      
      
      p = teacher_xx-teacher_x
      q = teacher_yy-teacher_y
      sum += abs(p)
      sum += abs(q)
      a.append(p)
      a.append(q)
      
      
 
 
 
 #print("len a = ", len(a))
 #print(sum)
 print("Avarage error rate is ",(sum/len(a)))
 #print("Avarage percentage error ",(sum2/len(a2)))
 


#dev()
#meanCorrelation()

test()