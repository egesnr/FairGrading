import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

'''
rows = []
with open("Implementation\Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)

'''
df = pd.read_csv("Grading_Assignment.csv")


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


# The inter relation data 25*25
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


# Given a data which has for each instructor's pairwise inter-relation
# As a mean difference with other instructors predicts not given instructor using sample data
# (sample data should be 2d array)
# returns predictions

def MLModelA_simple(inter_data, sample_data):
    for i in range(len(sample_data)):
        temp = []
        unknown_temp = []
        for a in range(len(sample_data[i])):
            if pd.notna(sample_data[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    corr = inter_data[temp[j]][unknown_temp[b]]
                    tempSum += sample_data[i][temp[j]]
                    tempSum += corr
                else:
                    continue
            average = tempSum / len(temp)
            if average >= 100:
                average = 100
            elif average <= 0:
                average = 0
            sample_data[i][unknown_temp[b]] = average

    return sample_data


# for each instructor's grades finds the frequency and returns 1D array
# first element of 1D array correspond to first column in data which is instructor one and so on.
def find_frequency(data):
    insGradeLen = []
    mySum = 0
    for i in range(len(data.columns)):
        column = data.iloc[:, [i]]
        column = column.dropna()
        mySum += len(column)
        insGradeLen.append(len(column))
    insGradeLen = [x / mySum for x in insGradeLen]
    return insGradeLen


# weighted randomized splitting data train and test by looking the frequency
# Returns trains 2d array and test dictionary source and grade as key and value correspondingly
def split_data(data):
    train_data = data.to_numpy().copy()
    freq = find_frequency(data)
    freq = pd.DataFrame(freq)
    locations = [[]]
    test_dict = [{}]

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            if pd.notna(train_data[i][j]):
                locations[i].append(j)
        locations.append([])
    locations = locations[:-1]

    for i in range(len(locations)):
        var = random.choices(locations[i], weights=np.concatenate(freq.iloc[locations[i]].to_numpy()))
        no = var[0]
        test_dict[i].update({no: train_data[i, no]})
        train_data[i][no] = None
        test_dict.append(dict())
    test_dict = test_dict[:-1]
    return train_data, test_dict


# calculates error array, root mean square error
def validation(predictions, test_data):
    error = []
    rmse = 0.0
    for i in range(len(test_data)):
        for key, value in test_data[i].items():
            rmse += ((predictions[i][key] - value) ** 2)
            error.append(abs(predictions[i][key] - value))

    rmse = math.sqrt(rmse / len(test_data))
    return error, rmse


def multi_split(k):
    error = 0.0
    RMSE = 0.0
    for i in range(k):
        train, test1 = split_data(df2)
        pre = MLModelA_simple(meanCorrelation(train), train)
        err, r = validation(pre, test1)
        print("Here is the mean of error rate: {0:.2f} [Max: {1:.2f}, Min: {2:.2f}]".format(np.mean(err), np.max(err),
                                                                                            np.min(err)))
        print("RMSE: " + str(r))
        print()
        error += np.mean(err)
        RMSE += r
    error = error / k
    RMSE = RMSE / k
    print("Here is the error mean {0:.2f} : ".format(error))
    print("RMSE : {0:.2f} ".format(RMSE))


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

    for j in range(i + 1, 25):
        combinations = df2.iloc[:, [i, j]]
        commons = combinations.dropna()
        length = len(commons)

        random_number1 = random.randint(0, length - 1)
        random_number2 = random.randint(0, length - 1)
        teacher_x = commons.iloc[random_number1, 0]
        teacher_y = commons.iloc[random_number1, 1]

        teacher_xx = commons.iloc[:, 0]
        teacher_yy = commons.iloc[:, 1]

        teacher_xx = teacher_xx.loc[teacher_xx != teacher_x].mean()
        teacher_yy = teacher_yy.loc[teacher_yy != teacher_y].mean()

        # Theorem 1
        p = teacher_xx - teacher_x
        q = teacher_yy - teacher_y
        sum += abs(p)
        sum += abs(q)
        a.append(p)
        a.append(q)
        # Bu kullanılacak !!
        mean_diff = teacher_xx - teacher_yy
        # düzeltilecek
        # predict_x = teacher_y + mean_diff
        # predict_y = teacher_x + mean_diff

        diff = teacher_xx - teacher_yy
        deviation_x = np.sqrt(np.sum((teacher_x - teacher_xx) ** 2) / length)
        deviation_y = np.sqrt(np.sum((teacher_y - teacher_yy) ** 2) / length)
        predict_x = teacher_y + mean_diff
        predict_y = teacher_x - mean_diff
        sum2 += abs(predict_x - teacher_x)
        sum2 += abs(predict_y - teacher_y)
        a2.append(predict_x)
        a2.append(predict_y)

        devide_diff = teacher_xx / teacher_yy
        predict_2x = teacher_y * devide_diff
        predict_2y = teacher_x * devide_diff
        sum3 += abs(predict_2x - teacher_x)
        sum3 += abs(predict_2y - teacher_y)
        a3.append(predict_2x)
        a3.append(predict_2y)
        # Theorem 4 kontrol et
        if (teacher_x - teacher_xx) > deviation_x:
            predict_3y = predict_2y
        else:
            predict_3y = teacher_yy
        if (teacher_y - teacher_yy) > deviation_y:
            predict_3x = predict_2x
        else:
            predict_3x = teacher_xx
        sum4 = abs(predict_3x - teacher_x)
        sum4 = abs(predict_3y - teacher_y)
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

# print("len a = ", len(a))
# print(sum)
print("Avarage error rate is ", (sum / len(a)))
print("Avarage error rate for theorem 2 is ", sum2 / len(a2))
print("Avarage error rate for theorem 3 is ", sum3 / len(a3))
print("Avarage error rate for theorem 4 is ", sum4 / len(a4))
# print("Avarage percentage error ",(sum2/len(a2)))


# dev()
# meanCorrelation_train()

multi_split(20)
