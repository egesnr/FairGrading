import statistics
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# Dropped ID Column for d2 dataframe
df2 = df.drop('ID', inplace=False, axis=1)


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


# This model finds every graded index and non-graded index, gets the average of all graded values for same project
# Then gives prediction for every other non-graded lecturer for same project
def MLModel_base(data):
    train = copy.deepcopy(data)

    for i in range(len(train)):
        temp = []
        unknown_temp = []
        temp_sum = 0
        for a in range(len(train[i])):
            if pd.notna(train[i][a]):
                temp.append(a)
                temp_sum += train[i][a]
            else:
                unknown_temp.append(a)

        temp_sum = temp_sum / len(temp)
        if temp_sum > 100:
            temp_sum = 100
        for b in range(len(unknown_temp)):
            train[i][unknown_temp[b]] = temp_sum
    return train


# The inter relation data 25*25, common project difference matrix
def meanCorrelation(train_data):
    a = []
    train_data = pd.DataFrame(train_data)
    for i in range(len(train_data.columns)):
        for j in range(len(train_data.columns)):
            combinations = train_data.iloc[:, [i, j]]
            commons = combinations.dropna()
            teacher_x = commons.iloc[:, 0].mean()
            teacher_y = commons.iloc[:, 1].mean()
            var = teacher_x - teacher_y
            if var > 0:
                a.append(math.ceil(var))
            else:
                a.append(math.floor(var))

    b = np.array(a)
    c = b.reshape(25, 25)
    d = pd.DataFrame(c)
    return d


def MLModel_(inter_data, sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    corr = inter_data[temp[j]][unknown_temp[b]]
                    tempSum += sample_data1[i][temp[j]]
                    tempSum += corr

                else:
                    continue
            average = tempSum / len(temp)
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


def MLModel_VariationC(inter_data, sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    corr_table = correlation_table(df2)
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    if corr_table[temp[j]][unknown_temp[b]] >= 0.55:
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        tempSum += corr

                else:
                    continue
            average = tempSum / len(temp)
            if average > 100:
                tempSum = 0
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum += corr
                        elif corr > 0:
                            tempSum += math.log(corr,1.5)
                    else:
                        continue
                average = tempSum / len(temp)
                if average > 100:
                    average = 100
            elif average < 0:
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum -= math.log(abs(corr),1.8)
                        elif corr > 0:
                            tempSum += corr
                    else:
                        continue
                average = tempSum / len(temp)
                if average < 0:
                    average = 0
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


def MLModel_VariationA(inter_data, sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    corr = inter_data[temp[j]][unknown_temp[b]]
                    tempSum += sample_data1[i][temp[j]]
                    tempSum += corr

                else:
                    continue
            average = tempSum / len(temp)
            if average > 100:
                average = 100
            elif average < 0:
                average = 0
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


# Given a data which has for each instructor's pairwise inter-relation
# As a mean difference with other instructors predicts not given instructor using sample data
# (sample data should be 2d array)
# returns predictions

def MLModel_VariationB(inter_data, sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    corr_table = correlation_table(df2)
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    if corr_table[temp[j]][unknown_temp[b]] >= 0.55:
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        tempSum += corr

                else:
                    continue
            average = tempSum / len(temp)
            if average > 100:
                average = 100
            elif average < 0:
                average = 0
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


# for each instructor's grades finds the frequency and returns 1D array
# first element of 1D array correspond to first column in data which is instructor one and so on.
def find_frequency(data):
    insGradeLen = []
    for i in range(len(data.columns)):
        column = data.iloc[:, [i]]
        column = column.dropna()
        insGradeLen.append(len(column))
    return insGradeLen


# Find how many common grade for each pair of teacher
def common_grade_table(data):
    freq = pd.DataFrame()
    freq2 = []
    for i in range(25):
        for j in range(25):
            combinations = data.iloc[:, [i, j]]
            commons = combinations.dropna()
            freq2.append(len(commons.iloc[:, 0]))
    b = np.array(freq2)
    c = b.reshape(25, 25)
    d = pd.DataFrame(c)
    return d


def split_data_randomized(data):
    train_data = copy.deepcopy(data.to_numpy())
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
        freq_teacher_1 = freq.iloc[locations[i][0]][0]
        freq_teacher_2 = freq.iloc[locations[i][1]][0]
        freq_teacher_3 = freq.iloc[locations[i][2]][0]

        var = random.choices(locations[i], weights=[freq_teacher_1, freq_teacher_2, freq_teacher_3])
        no = var[0]
        if no == locations[i][0]:
            freq.iloc[locations[i][0]][0] = freq_teacher_1 - 1
        elif no == locations[i][1]:
            freq.iloc[locations[i][1]][0] = freq_teacher_2 - 1
        elif no == locations[i][2]:
            freq.iloc[locations[i][2]][0] = freq_teacher_3 - 1
        test_dict[i].update({no: train_data[i, no]})
        train_data[i][no] = None
        test_dict.append(dict())
    test_dict = test_dict[:-1]
    return train_data, test_dict


def split_data(data, x):
    global var
    train_data = copy.deepcopy(data.to_numpy())
    freq = find_frequency(pd.DataFrame(train_data))
    freq = pd.DataFrame(freq)
    commons = common_grade_table(pd.DataFrame(train_data))
    locations = [[]]
    test_dict = [{}]

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            if pd.notna(train_data[i][j]):
                locations[i].append(j)
        locations.append([])
    locations = locations[:-1]

    for i in range(len(locations)):
        teacher_1 = locations[i][0]
        teacher_2 = locations[i][1]
        teacher_3 = locations[i][2]

        freq_teacher_1 = freq.iloc[locations[i][0]][0]
        freq_teacher_2 = freq.iloc[locations[i][1]][0]
        freq_teacher_3 = freq.iloc[locations[i][2]][0]

        commons_per_project2 = {0: int(commons[teacher_1][teacher_2]), 1: int(commons[teacher_1][teacher_3]),
                                2: int(commons[teacher_2][teacher_3])
                                }
        # Sorting dict

        commons_per_project2 = {k: v for k, v in sorted(commons_per_project2.items(), key=lambda item: item[1])}

        # get last item which has max value pair
        if list(commons_per_project2)[-1] == 0:

            if commons[teacher_1][teacher_2] >= x:

                if freq_teacher_1 >= freq_teacher_2:
                    var = teacher_1
                    freq.iloc[locations[i][0]][0] = freq_teacher_1 - 1
                else:
                    var = teacher_2
                    freq.iloc[locations[i][1]][0] = freq_teacher_2 - 1
                commons[teacher_1][teacher_2] -= 1
                commons[teacher_2][teacher_1] -= 1

        elif list(commons_per_project2)[-1] == 1:

            if commons[teacher_1][teacher_3] >= x:

                if freq_teacher_1 >= freq_teacher_3:
                    var = teacher_1
                    freq.iloc[locations[i][0]][0] = freq_teacher_1 - 1
                else:
                    var = teacher_3
                    freq.iloc[locations[i][2]][0] = freq_teacher_3 - 1
                commons[teacher_1][teacher_3] -= 1
                commons[teacher_3][teacher_1] -= 1


        elif list(commons_per_project2)[-1] == 2:

            if commons[teacher_2][teacher_3] >= x:

                if freq_teacher_2 >= freq_teacher_3:
                    freq.iloc[locations[i][1]][0] = freq_teacher_2 - 1
                    var = teacher_2
                else:
                    var = teacher_3
                    freq.iloc[locations[i][2]][0] = freq_teacher_3 - 1
                commons[teacher_2][teacher_3] -= 1
                commons[teacher_3][teacher_2] -= 1

        test_dict[i] = {var: train_data[i, var]}

        test_dict.append({})
        train_data[i][var] = None

    test_dict = test_dict[:-1]

    return train_data, test_dict


# calculates error array, root mean square error
def validation(predictions, test_data):
    error = []
    rmse = 0.0
    for i in range(len(test_data)):
        for key, value in test_data[i].items():
            if np.isnan(value) == False:
                rmse += ((predictions[i][key] - value) ** 2)

                error.append(abs(predictions[i][key] - value))

    print("Length ", len(error))
    rmse = math.sqrt(rmse / len(test_data))

    return error, rmse


def validation_source_truth(predictions):
    truth = pd.read_csv("Truth.csv")
    truth2 = truth.drop('Project ID', inplace=False, axis=1)
    truth2 = truth2.T.to_numpy().flatten()

    avg_p = []
    rmse = 0
    error = []
    # calculate the mean for all projects
    for i in range(len(predictions)):
        avg_p.append(np.mean(predictions[i]))
    # Calculates RMSE
    for i in range(len(avg_p)):
        error.append(abs(avg_p[i] - truth2[i]))
        rmse += ((avg_p[i] - truth2[i]) ** 2)

    rmse = math.sqrt(rmse / len(avg_p))
    return error, rmse


def multi_split(a, b):
    error = 0.0
    RMSE = 0.0
    for i in range(a, b):
        train, test1 = split_data(df2, i)
        pre = MLModel_VariationB(meanCorrelation(train), train)
        err, r = validation(pre, test1)
        print("This is for " + str(i))
        print("Here is the mean of error rate: {0:.2f} [Max: {1:.2f}, Min: {2:.2f}]".format(np.mean(err), np.max(err),
                                                                                            np.min(err)))
        print("RMSE: " + str(r))
        print()
        error += np.mean(err)
        RMSE += r
    error = error / (b - a)
    RMSE = RMSE / (b - a)
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


def standart_deviation_table(data):
    standart_deviation_table = []
    for i in range(25):
        for j in range(25):
            combinations = data.iloc[:, [i, j]]
            combinations_dropped = combinations.dropna()
            first_column = combinations_dropped.iloc[:, 0]
            second_column = combinations_dropped.iloc[:, 1]
            column_diffrence = first_column - second_column
            # population standard deviation
            standart_deviation_table.append(statistics.pstdev(column_diffrence))

    standart_deviation_table = np.array(standart_deviation_table)
    standart_deviation_table = standart_deviation_table.reshape(25, 25)
    return pd.DataFrame(standart_deviation_table)


def correlation_table(data):
    corr_table = []
    for i in range(25):
        for j in range(25):
            combinations = data.iloc[:, [i, j]]
            combinations_dropped = combinations.dropna()
            first_column = combinations_dropped.iloc[:, 0]
            second_column = combinations_dropped.iloc[:, 1]
            column_diffrence = first_column - second_column

            corr_table.append(statistics.correlation(first_column, second_column))

    corr_table = np.array(corr_table)
    corr_table = corr_table.reshape(25, 25)
    return pd.DataFrame(corr_table)


# control function
def first_optimization(split_value):
    train, test = split_data(df2, split_value)
    train = MLModel_base(train)
    err, rmse = validation(train, test)

    print("RMSE value: " + str(rmse))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()
    train2 = MLModel_base(df2.to_numpy())
    err2, rmse2 = validation_source_truth(train2)
    print("RMSE value: " + str(rmse2))
    print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    print("Error average:" + str(np.mean(err2)))


# Control function for MLmodel_VariationA
def second_optimization(split_value):
    train, test = split_data(df2, split_value)
    train = MLModel_(meanCorrelation(train), train)
    err, rmse = validation(train, test)
    print("RMSE value: " + str(rmse))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()
    train2 = MLModel_(meanCorrelation(df2.to_numpy()), df2.to_numpy())
    err2, rmse2 = validation_source_truth(train2)
    print("RMSE value: " + str(rmse2))
    print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    print("Error average:" + str(np.mean(err2)))


def third_optimization(split_value):
    train, test = split_data(df2, split_value)
    train = MLModel_VariationA(meanCorrelation(train), train)
    err, rmse = validation(train, test)
    print("RMSE value: " + str(rmse))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()
    train2 = MLModel_VariationA(meanCorrelation(df2.to_numpy()), df2.to_numpy())
    err2, rmse2 = validation_source_truth(train2)
    print("RMSE value: " + str(rmse2))
    print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    print("Error average:" + str(np.mean(err2)))


def fourth_optimization(split_value):
    train, test = split_data(df2, split_value)
    meanCorrelation(train).to_csv("differ_aSplit.csv")
    train = MLModel_VariationB(meanCorrelation(train), train)
    err, rmse = validation(train, test)
    print("RMSE value: " + str(rmse))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()
    train2 = MLModel_VariationB(meanCorrelation(df2.to_numpy()), df2.to_numpy())
    err2, rmse2 = validation_source_truth(train2)
    print("RMSE value: " + str(rmse2))
    print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    print("Error average:" + str(np.mean(err2)))


def fifth_optimization(split_value):
    train, test = split_data(df2, split_value)
    meanCorrelation(train).to_csv("differ_aSplit.csv")
    train = MLModel_VariationC(meanCorrelation(train), train)
    err, rmse = validation(train, test)
    print("RMSE value: " + str(rmse))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()
    train2 = MLModel_VariationC(meanCorrelation(df2.to_numpy()), df2.to_numpy())
    err2, rmse2 = validation_source_truth(train2)
    print("RMSE value: " + str(rmse2))
    print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    print("Error average:" + str(np.mean(err2)))


# first_optimization(10)
# second_optimization(10)
# third_optimization(10)
# fourth_optimization(10)
fifth_optimization(10)
