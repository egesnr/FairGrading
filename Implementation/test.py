import statistics
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor

'''
rows = []
with open("Implementation\Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)

'''
df = pd.read_csv("PIEGrammaticalData.csv")
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
    # plt.hist(teacher1)
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

    deviation = np.array(deviation).reshape(4, 4)
    deviation = pd.DataFrame(deviation)


# This model finds every graded index and non-graded index, gets the average of all graded values for same project
# Then gives prediction for every other non-graded lecturer for same project
def MLModel_base(data, clampMax, clampMin):
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
        if temp_sum > clampMax:
            temp_sum = clampMax
        elif temp_sum < clampMin:
            temp_sum = clampMin
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
            if np.isnan(var):
                var = 0
            # TODO: burayı bir konuşalım
            if var > 0:
                a.append(math.ceil(var))
            else:
                a.append(math.floor(var))

    b = np.array(a)
    c = b.reshape(4, 4)
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


def MLModel_VariationC(inter_data, sample_data):
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
            if average > 100:  # 150
                # x/ x-1 f(x) = aktivasyon
                # average aldıktan sonra logaritmik bir fonsiyona koy onun sonucu yaz
                tempSum = 0
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum += corr
                        elif corr > 0:
                            tempSum += math.log(corr, 1.5)
                    else:
                        continue
                average = tempSum / len(temp)

            elif average < 0:
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum -= math.log(abs(corr), 1.8)
                        elif corr > 0:
                            tempSum += corr
                    else:
                        continue
                average = tempSum / len(temp)
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


def MLModel_VariationD(inter_data, sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    sdt_of_pairs = standart_deviation_table(pd.DataFrame(df2))
    cosines = cosine_similarity(pd.DataFrame(df2))
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []

        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            sdt_array = []
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    tempVar = 0
                    corr = inter_data[temp[j]][unknown_temp[b]]
                    tempVar += sample_data1[i][temp[j]]
                    tempVar += corr
                    tempVar = tempVar * (1 / std_of_individuals[temp[j]][unknown_temp[b]])
                    tempSum += tempVar
                    sdt_array.append(1 / std_of_individuals)
                else:
                    continue
            divided = sum(sdt_array)
            average = tempSum / sum(sdt_array)
            if average > 100:  # 150
                # x/ x-1 f(x) = aktivasyon
                # average aldıktan sonra logaritmik bir fonsiyona koy onun sonucu yaz
                tempSum = 0
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum += corr
                        elif corr > 0:
                            tempSum += math.log(corr, 1.5)
                    else:
                        continue
                average = tempSum / len(temp)

            elif average < 0:
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum -= math.log(abs(corr), 1.8)
                        elif corr > 0:
                            tempSum += corr
                    else:
                        continue
                average = tempSum / len(temp)
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


def MLModel_VariationE(inter_data, sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    sdt_of_pairs = standart_deviation_table(pd.DataFrame(df2))
    cosines = cosine_similarity(pd.DataFrame(df2))
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []

        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        for b in range(len(unknown_temp)):
            sdt_array = []
            tempSum = 0
            for j in range(len(temp)):
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                    tempVar = 0
                    corr = inter_data[temp[j]][unknown_temp[b]]
                    tempVar += sample_data1[i][temp[j]]
                    tempVar += corr
                    tempVar = tempVar * (1 / std_of_individuals[temp[j]][unknown_temp[b]])
                    tempSum += tempVar
                    sdt_array.append(1 / std_of_individuals)
                    sdt_array.append(1/sdt_of_pairs[temp[j]][unknown_temp[b]])
                else:
                    continue
            divided = sum(sdt_array)
            average = tempSum / sum(sdt_array)
            if average > 100:  # 150
                # x/ x-1 f(x) = aktivasyon
                # average aldıktan sonra logaritmik bir fonsiyona koy onun sonucu yaz
                tempSum = 0
                for j in range(len(temp)):
                    if pd.notna(inter_data[temp[j]][unknown_temp[b]]):
                        corr = inter_data[temp[j]][unknown_temp[b]]
                        tempSum += sample_data1[i][temp[j]]
                        if corr < 0:
                            tempSum += corr
                        elif corr > 0:
                            tempSum += math.log(corr, 1.5)
                    else:
                        continue
                average = tempSum / len(temp)
            average = tempSum / divided
            if average > 100:
                average = 100

            elif average < 0:
                average = 0

            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


def MLModel_collaborative(inter_data, sample_data, clampMax, clampMin, graderNo):
    sample_data1 = copy.deepcopy(sample_data)
    # Burda niye df2 kullanıyoruz çünkü veri ayırma yapınca yeteri kadar ortak veri bulamıyoruz ve hata veriyor
    corr_table = correlation_table(pd.DataFrame(sample_data1), graderNo)
    common_table = common_grade_table(pd.DataFrame(df2))
    general_mean_table = take_the_bias(df2.to_numpy())
    sdt = standart_deviation_table(pd.DataFrame(sample_data))
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

            sample_data1_df = pd.DataFrame(sample_data1)
            corrSum = 0
            for j in range(len(temp)):
                # check to if there is no relation between some pairs
                if pd.notna(inter_data[temp[j]][unknown_temp[b]]):

                    common_diff = inter_data[temp[j]][unknown_temp[b]]

                    adjusted = sample_data[i][temp[j]] + common_diff

                    # Pearson's Correlation
                    corr = corr_table[temp[j]][unknown_temp[b]]

                    weighted = adjusted * corr

                    corrSum += abs(corr)
                    tempSum += weighted

                    common_diff = general_mean_table[unknown_temp[b]] - general_mean_table[temp[j]]
                    adjusted = sample_data[i][temp[j]] + common_diff

                    # Pearson's Correlation
                    corr = corr_table[temp[j]][unknown_temp[b]]

                    weighted = adjusted * abs(corr)

                    corrSum += abs(corr)
                    tempSum += weighted

                else:
                    continue
            average = tempSum / corrSum
            #TODO: it will be updated
            if average > clampMax:
                average = clampMax
            elif average < clampMin:
                average = clampMin
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1


def MLModel_cumulative(individual_data, sample_data, clampMax, clampMin):
    sample_data1 = copy.deepcopy(sample_data)
    # Burda niye df2 kullanıyoruz çünkü veri ayırma yapınca yeteri kadar ortak veri bulamıyoruz ve hata veriyor
    corr_table = meanCorrelation(df2.to_numpy())
    common_table = common_grade_table(pd.DataFrame(df2))
    general_mean_table = take_the_bias(df2.to_numpy())
    sdt = std_of_individuals(pd.DataFrame(df2))
    sdt2 = standart_deviation_table(pd.DataFrame(df2))
    y_hat = []

    for i in range(len(sample_data1)):
        temp = []
        unknown_temp = []
        scores = []
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)

        tempSum = 0
        count = 0
        for j in range(len(temp)):
            common_diff = general_mean_table[temp[j]]
            adjusted = sample_data1[i][temp[j]] - common_diff
            row_except_element = temp[:j] + temp[j + 1:]
            raw_score = sample_data1[i][temp[j]] - (sum(sample_data1[i][row_except_element]) / 2)

            score_z = (raw_score - general_mean_table[temp[j]])/sdt[temp[j]]
            #tempSum += adjusted*(1/sdt[temp[j]])
            tempSum += (sample_data1[i][temp[j]] - common_diff)*(1/(sdt[temp[j]]))
            count += 1/(sdt[temp[j]])
            '''
            score_x = 1/(sdt[temp[j]]+1)
            score_p = 1
            coefficient = 1/(score_z+0.1)
            coefficient2 = common_diff*score_z        
                    
            weighted =  sample_data1[i][temp[j]] - coefficient2
            scores.append(coefficient)
                   
            tempSum += weighted
            '''
                 
                
        average = tempSum / count

        if average > clampMax:
            average = clampMax
        elif average < clampMin:
            average = clampMin
        y_hat.append(average)

    return y_hat


def NN(sample_data, inter_data):
    data = np.array([])
    sample_data1 = copy.deepcopy(sample_data)
    corr_table = correlation_table(df2)
    corr_table.to_csv("csv3.csv")
    for i in range(len(sample_data)):
        temp = []
        temp2 = []
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
                temp2.append(sample_data[i][a])
        temp2.append(inter_data[temp[0]][temp[1]])
        temp2.append(inter_data[temp[0]][temp[2]])
        temp2.append(inter_data[temp[1]][temp[2]])
        temp2.append(corr_table[temp[0]][temp[1]])
        temp2.append(corr_table[temp[0]][temp[2]])
        temp2.append(corr_table[temp[1]][temp[2]])

        data = np.append(data, temp2)
    data = data.reshape(1000, 9)

    y = data[:, 0]

    data = data[:, 1:]
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # test_pre = lr.predict(y_test)
    train_pre = lr.predict(X_test)
    print(abs(y_test - train_pre).mean())
    print("Mean squared error: %.2f" % mean_squared_error(y_test, train_pre))

    model_SVR = svm.SVR()
    model_SVR.fit(X_train, y_train)
    Y_pred = model_SVR.predict(X_test)
    print(abs(y_test - Y_pred).mean())
    print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred))

    model_RFR = RandomForestRegressor(n_estimators=10)
    model_RFR.fit(X_train, y_train)
    Y_predd = model_RFR.predict(X_test)
    print(abs(y_test - Y_predd).mean())
    print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_predd))


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
# Range check
def common_grade_table(data,):
    freq2 = []
    for i in range(4):
        for j in range(4):
            combinations = data.iloc[:, [i, j]]
            commons = combinations.dropna()
            freq2.append(len(commons.iloc[:, 0]))
    b = np.array(freq2)
    c = b.reshape(4, 4)
    d = pd.DataFrame(c)
    return d


from numpy.linalg import norm


def cosine_similarity(data):
    cosine_table = []
    for i in range(25):
        for j in range(25):
            combinations = data.iloc[:, [i, j]]
            combinations_dropped = combinations.dropna()
            first_column = combinations_dropped.iloc[:, 0]
            second_column = combinations_dropped.iloc[:, 1]

            cosine = np.dot(first_column, second_column) / (norm(first_column) * norm(second_column))

            # population standard deviation
            cosine_table.append(cosine)

    cosine_table = np.array(cosine_table)
    cosine_table = cosine_table.reshape(25, 25)
    return cosine_table


def take_the_bias(data):
    array = []
    data = pd.DataFrame(data)
    # column index

    for i in data.columns:
        ex = data[data.loc[:, i].isnull() == False]
        # row index
        s = 0
        d = 0
        for j in ex.index:
            # others column index
            for k in ex.columns:
                if i == k or np.isnan(ex.at[j, k]):
                    continue
                s += ex.at[j, i] - ex.at[j, k]
                d += 1
        array.append(s / d)

    return array


def std_of_individuals(data):
    array = []

    data = pd.DataFrame(data)
    # column index

    for i in data.columns:
        array_2 = []
        ex = data[data.loc[:, i].isnull() == False]
        # row index
        s = 0
        d = 0
        for j in ex.index:
            x = 0
            y = 0
            # others column index
            for k in ex.columns:
                if i == k or np.isnan(ex.at[j, k]):
                    continue
                x += ex.at[j, i] - ex.at[j, k]
                y += 1
                array_2.append(x / y)
        std = statistics.pstdev(array_2)
        array.append(std)
    return array


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
    # burda hocalarıda soft code şekilde alman lazım her hoca sayısı için uymalı
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


def split_data(data, frequency_limit):
    var = 0
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

            if commons[teacher_1][teacher_2] >= frequency_limit:

                if commons[teacher_1][teacher_3] >= commons[teacher_2][teacher_3]:
                    var = teacher_1
                    # freq.iloc[locations[i][0]][0] = freq_teacher_1 - 1
                    commons[teacher_1][teacher_2] -= 1
                    commons[teacher_1][teacher_3] -= 1

                    commons[teacher_2][teacher_1] -= 1
                    commons[teacher_3][teacher_1] -= 1
                else:
                    var = teacher_2
                    # freq.iloc[locations[i][1]][0] = freq_teacher_2 - 1
                    commons[teacher_2][teacher_1] -= 1
                    commons[teacher_2][teacher_3] -= 1

                    commons[teacher_1][teacher_2] -= 1
                    commons[teacher_3][teacher_2] -= 1

        elif list(commons_per_project2)[-1] == 1:

            if commons[teacher_1][teacher_3] >= frequency_limit:

                if commons[teacher_1][teacher_2] >= commons[teacher_3][teacher_2]:
                    var = teacher_1
                    # freq.iloc[locations[i][0]][0] = freq_teacher_1 - 1
                    commons[teacher_1][teacher_2] -= 1
                    commons[teacher_1][teacher_3] -= 1

                    commons[teacher_2][teacher_1] -= 1
                    commons[teacher_3][teacher_1] -= 1
                else:
                    var = teacher_3
                    # freq.iloc[locations[i][2]][0] = freq_teacher_3 - 1
                    commons[teacher_3][teacher_1] -= 1
                    commons[teacher_3][teacher_2] -= 1

                    commons[teacher_1][teacher_3] -= 1
                    commons[teacher_2][teacher_3] -= 1


        elif list(commons_per_project2)[-1] == 2:

            if commons[teacher_2][teacher_3] >= frequency_limit:

                if commons[teacher_2][teacher_1] >= commons[teacher_3][teacher_1]:
                    # freq.iloc[locations[i][1]][0] = freq_teacher_2 - 1
                    var = teacher_2
                    commons[teacher_2][teacher_3] -= 1
                    commons[teacher_2][teacher_1] -= 1

                    commons[teacher_3][teacher_2] -= 1
                    commons[teacher_1][teacher_2] -= 1
                else:
                    var = teacher_3
                    # freq.iloc[locations[i][2]][0] = freq_teacher_3 - 1
                    commons[teacher_3][teacher_1] -= 1
                    commons[teacher_3][teacher_2] -= 1

                    commons[teacher_1][teacher_3] -= 1
                    commons[teacher_2][teacher_3] -= 1


        test_dict[i] = {var: train_data[i, var]}

        test_dict.append({})
        train_data[i][var] = None

    test_dict = test_dict[:-1]

    return train_data, test_dict


# calculates error array, root mean square error
def validation(predictions, test_data, data):
    error = []
    rmse = 0.0
    rmse_average = 0
    index = data.notna()
    for x in range(len(predictions)):
        for y in range(len(predictions[x])):
            if not index.iloc[x][y]:
                predictions[x][y] = np.nan
    for i in range(len(test_data)):
        average = data.iloc[i][0: 4].mean()
        yhat = np.nanmean(predictions[i][0: 4])
        rmse_average += (yhat - average) ** 2
        for key, value in test_data[i].items():
            if np.isnan(value) == False:
                rmse += ((predictions[i][key] - value) ** 2)

                error.append(abs(predictions[i][key] - value))
                predictions[i][key] = np.nan

    print("Length ", len(error))
    rmse = math.sqrt(rmse / len(test_data))
    rmse_average =  math.sqrt(rmse_average / len(test_data))
    return error, rmse, rmse_average

def validation_cumulative(yhat, data):
    rmse = 0
    for i in range(len(yhat)):

        average = data.iloc[i][0: 4].sum() / 3
        rmse += (average - yhat[i]) ** 2

    return math.sqrt(rmse / len(yhat))

def take_the_bias(data):
    array = []
    data = pd.DataFrame(data)
    # column index

    for i in data.columns:
        ex = data[data.loc[:, i].isnull() == False]
        # row index
        s = 0
        d = 0
        for j in ex.index:
            # others column index
            for k in ex.columns:
                if i == k or np.isnan(ex.at[j, k]):
                    continue
                s += ex.at[j, i] - ex.at[j, k]
                d += 1
        array.append(s / d)

    return array



def validation_source_truth(predictions):
    truth = pd.read_csv("Implementation/Truth.csv")
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
    for i in range(4):
        for j in range(4):
            combinations = df2.iloc[:, [i, j]]
            commons = combinations.dropna()
            teacher_x = commons.iloc[:-1, 0].mean()
            teacher_y = commons.iloc[:-1, 1].mean()
            a.append(teacher_x - teacher_y)

    b = np.array(a)
    c = b.reshape(4, 4)

    d = pd.DataFrame(c)

    return d


def standart_deviation_table(data):
    standart_deviation_table = []
    for i in range(4):
        for j in range(4):
            combinations = data.iloc[:, [i, j]]
            combinations_dropped = combinations.dropna()
            first_column = combinations_dropped.iloc[:, 0]
            second_column = combinations_dropped.iloc[:, 1]
            column_diffrence = first_column - second_column
            # population standard deviation
            standart_deviation_table.append(statistics.pstdev(column_diffrence))

    standart_deviation_table = np.array(standart_deviation_table)
    standart_deviation_table = standart_deviation_table.reshape(4, 4)
    return pd.DataFrame(standart_deviation_table)


def correlation_table(data, graderNo):
    corr_table = []
    for i in range(graderNo):
        for j in range(graderNo):
            combinations = data.iloc[:, [i, j]]
            combinations_dropped = combinations.dropna()
            first_column = combinations_dropped.iloc[:, 0]
            second_column = combinations_dropped.iloc[:, 1]
            column_diffrence = first_column - second_column

            corr_table.append(statistics.correlation(first_column, second_column))

    corr_table = np.array(corr_table)
    corr_table = corr_table.reshape(graderNo, graderNo)
    return pd.DataFrame(corr_table)


# control function
def first_optimization(split_value):  # split value means that we are getting frequency of common projects
    train, test = split_data(df2, split_value)
    train = MLModel_base(train, 5, 0)
    err, rmse, rmse_a = validation(train, test, df2)
    print("validation score")
    print("RMSE value: " + str(rmse))
    print("RMSE_AVG value: " + str(rmse_a))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()



def second_optimization(split_value):
    train, test = split_data(df2, split_value)
    train = MLModel_(meanCorrelation(train), train)
    err, rmse = validation(train, test)
    print("RMSE value: " + str(rmse))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()
    train2 = MLModel_(meanCorrelation(df2.to_numpy()), df2.to_numpy())
    train2 = MLModel_(meanCorrelation(df2.to_numpy()), df2.to_numpy())
    err2, rmse2 = validation_source_truth(train2)
    print("RMSE value: " + str(rmse2))
    print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    print("Error average:" + str(np.mean(err2)))


# Control function for MLModel_VariationA
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


def sixth_optimization(split_value):
    train, test = split_data(df2, split_value)
    train = MLModel_collaborative(meanCorrelation(train), train, 5, 0, 4)
    err, rmse, rmse_a = validation(train, test, df2)
    print("RMSE value: " + str(rmse))
    print("RMSE_AVG value: " + str(rmse_a))
    print("Max & Min : " + str(max(err)) + " & " + str(min(err)))
    print("Error average: " + str(np.mean(err)))
    print()

    # print("randomized validation score ")
    # train2, test2 = split_data_randomized(df2)
    # train2 = MLModel_collaborative(meanCorrelation(train2), train2)
    # err2, rmse2 = validation(train2, test2)
    # print()
    # print("RMSE value: " + str(rmse2))
    # print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
    # print("Error average:" + str(np.mean(err2)))
    # print("Source of truth validation score")

def seventh_optimization(split_value):
    train, test = split_data(df2, split_value)
    train = MLModel_cumulative(meanCorrelation(train), train, 5, 0)
    rmse = validation_cumulative(train, df2)
    print("RMSE value: " + str(rmse))
    print()

first_optimization(10)
# second_optimization(10)
# third_optimization(10)
# fourth_optimization(10)
# fifth_optimization(10)
sixth_optimization(10)
seventh_optimization(10)

correlation_table(df2, 4)
