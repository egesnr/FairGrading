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
df = pd.read_csv("Implementation\Grading_Assignment.csv")


# df = pd.read_csv("Implementation\Grading_Assignment.csv")


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
            deviation.append(np.sqrt(np.sum((diff - mean) ** 2) / len(diff)))

    deviation = np.array(deviation).reshape(25, 25)
    deviation = pd.DataFrame(deviation)


# Use all data for training
def meanCorrelation():
    a = []
    for i in range(25):
        for j in range(25):
            combinations = df2.iloc[:, [i, j]]
            commons = combinations.dropna()
            teacher_x = commons.iloc[:, 0].mean()
            teacher_y = commons.iloc[:, 1].mean()
            a.append(teacher_x - teacher_y)

    b = np.array(a)
    c = b.reshape(25, 25)

    d = pd.DataFrame(c)
    return d


# Given a data which has for each instructor's pairwise inter-relation
# As a mean difference with other instructors predicts not given instructor using test data
# (test data should be represent grades with their source numbers)
# test data should be an array of dictionaries of int
# an example : [{1: None, 2: 63, 3: 73}, {3: None, 2: 76, 7: 90}]
#
def MLModelA_simple(inter_data, test_data):
    for i in range(len(test_data)):
        temp = []
        tempSum = 0
        unknown_temp = []
        for key in test_data[i].keys():
            if test_data[i][key] is not None:
                temp.append(key)
                tempSum += test_data[i][key]
            else:
                unknown_temp.append(key)

        for b in range(len(unknown_temp)):
            sumSum = tempSum
            for j in range(len(temp)):
                corr = inter_data[unknown_temp[b] - 1][temp[j] - 1]
                sumSum += corr
            average = sumSum / len(temp)
            if average >= 100:
                average = 100
            elif average <= 0:
                average = 0
            test_data[i][unknown_temp[b]] = int(average)


# given data of 2d int turns into dictionary
def conf_dictionary(data):
    return

# Throw the last item and train them
def meanCorrelation_train():
    a = []
    for i in range(25):
        for j in range(25):
            combinations = df2.iloc[:, [i, j]]
            commons = combinations.dropna()
            teacher_x = commons.iloc[:-1, 0].mean()
            teacher_y = commons.iloc[:-1, 1].mean()
            a.append(teacher_x - teacher_y)

    b = np.array(a)
    c = b.reshape(25, 25)

    d = pd.DataFrame(c)

    return d



a = []
a2 = []
a3 = []
a4 = []
deviation = []
sum = 0
sum2 = 0
sum3 = 0
sum4 = 0
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
      teacher_y = commons.iloc[random_number1,1]

      teacher_xx = commons.iloc[:,0]
      teacher_yy = commons.iloc[:,1]
      
      teacher_xx = teacher_xx.loc[teacher_xx!=teacher_x].mean()
      teacher_yy = teacher_yy.loc[teacher_yy!=teacher_y].mean()
     
      #Theorem 1
      p = teacher_xx-teacher_x
      q = teacher_yy-teacher_y
      sum += abs(p)
      sum += abs(q)
      a.append(p)
      a.append(q)
     #Bu kullanılacak !!
      mean_diff = teacher_xx - teacher_yy
     #düzeltilecek
      #predict_x = teacher_y + mean_diff
      #predict_y = teacher_x + mean_diff  
      
      diff = teacher_xx - teacher_yy
      deviation_x = np.sqrt(np.sum((teacher_x - teacher_xx) ** 2) /length)
      deviation_y = np.sqrt(np.sum((teacher_y - teacher_yy) ** 2) /length)
      predict_x = teacher_y + mean_diff
      predict_y = teacher_x - mean_diff
      sum2 += abs(predict_x-teacher_x)
      sum2 += abs(predict_y-teacher_y)
      a2.append(predict_x)
      a2.append(predict_y)

      devide_diff = teacher_xx/teacher_yy
      predict_2x = teacher_y*devide_diff
      predict_2y = teacher_x*devide_diff
      sum3 += abs(predict_2x-teacher_x)
      sum3 += abs(predict_2y-teacher_y)
      a3.append(predict_2x)
      a3.append(predict_2y)
      #Theorem 4 kontrol et 
      if (teacher_x - teacher_xx)>deviation_x:
        predict_3y = predict_2y
      else:
        predict_3y = teacher_yy
      if (teacher_y - teacher_yy)>deviation_y:
        predict_3x = predict_2x
      else:
        predict_3x = teacher_xx
      sum4 = abs(predict_3x-teacher_x)
      sum4 = abs(predict_3y-teacher_y)
      a4.append(predict_3x)
      a4.append(predict_3y)
      '''
      if abs(teacher_x-teacher_xx)>deviation_x:
         predict_y = teacher_x + mean_diff + (abs(teacher_x-teacher_xx)-deviation_x)
      else:
         predict_y = teacher_x + mean_diff - (abs(teacher_x-teacher_xx)-deviation_x)
      if  abs(teacher_y-teacher_yy)>deviation_y:
         predict_x = teacher_y + mean_diff + (abs(teacher_y-teacher_yy)-deviation_y)
      else:
         predict_x = teacher_y + mean_diff - + (abs(teacher_y-teacher_yy)-deviation_y)
    
      value_one = predict_x - teacher_x
      value_two = predict_y - teacher_y
      sum2 += abs(value_one)
      sum2 += abs(value_two)
      a2.append(value_one)
      a2.append(value_two)
      '''
     
      
      
 
 
 
 #print("len a = ", len(a))
 #print(sum)
print("Avarage error rate is ",(sum/len(a)))
print("Avarage error rate for theorem 2 is ",sum2/len(a2))
print("Avarage error rate for theorem 3 is ",sum3/len(a3))
print("Avarage error rate for theorem 4 is ",sum4/len(a4))
 #print("Avarage percentage error ",(sum2/len(a2)))
 


#dev()
#meanCorrelation()


