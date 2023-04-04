import statistics
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import tensorflow as tf
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.decomposition import NMF


'''
rows = []
with open("Implementation\Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)

'''
df = pd.read_csv("Implementation\Grading_Assignment.csv")
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


def NN(sample_data,inter_data):
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
          

         
          data = np.append(data,temp2)
    data = data.reshape(1000,6)
    
    y_other = pd.read_csv('C:/Users/Ege/Desktop/FairGrading-1/Implementation/Truth.csv')
    y_other.drop('Project ID',inplace=True, axis = 1)
    
    y_other = np.array(y_other)
    y_other = y_other.reshape(-1)
    y = data[:,0]
    
    #data = data[:,1:]
    #print(y)
    X_train,X_test,y_train,y_test = train_test_split(data,y_other,test_size = 0.2,random_state= 0)
    lr = LinearRegression()
    lr.fit(X_train,y_train)

    #test_pre = lr.predict(y_test)
    train_pre = lr.predict(X_test)
    print('Error rate for Linear Regression %.2f'%abs(y_test - train_pre).mean())
    print("Mean squared error for Linear Regression: %.2f" % mean_squared_error(y_test, train_pre),'\n')
    
    model_SVR = svm.SVR()
    model_SVR.fit(X_train,y_train)
    Y_pred = model_SVR.predict(X_test)
    print('Error rate for SVR %.2f'%abs(y_test - Y_pred).mean())
    print("Mean squared error for SVR: %.2f" % mean_squared_error(y_test, Y_pred),"\n")
    
    model_RFR = RandomForestRegressor(n_estimators=10)
    model_RFR.fit(X_train, y_train)
    Y_predd = model_RFR.predict(X_test)
    print('Error rate for Random Forest Regression %.2f'%abs(y_test - Y_predd).mean())
    print("Mean squared error for Random Forest Regression: %.2f" % mean_squared_error(y_test, Y_predd),'\n')
    
    model_Decision_Tree =  DecisionTreeRegressor(max_depth=10)
    model_Decision_Tree.fit(X_train, y_train)
    Y_pred3 = model_Decision_Tree.predict(X_test)
    print('Error rate for Decision Tree regression %.2f'%abs(y_test - Y_pred3).mean())
    print("Mean squared error for Decision Tree regression: %.2f" % mean_squared_error(y_test, Y_pred3),'\n')

    ######################

    X = sample_data
    mean = np.nanmean(X)
    X[np.isnan(X)] = mean

    rank = 7
    model = NMF(n_components=rank, init='random', random_state=0, max_iter = 2500)

    # fit model to the matrix
    model.fit(X)

    # predict missing values
    W = model.transform(X)
    H = model.components_
    X_pred = np.dot(W, H)
   
    print("Predicted matrix:\n", X_pred)
          
def collaborative(sample_data):
    sample_data1 = copy.deepcopy(sample_data)
    corr_table = correlation_table(df2)
    corr_table.to_csv("csv3.csv")
    for i in range(len(sample_data1)):
        temp = []
        unknown_temp =[]
        teacher_list = []
        teacher_dic= {'one':0,'two':0,'three':0}
        for a in range(len(sample_data1[i])):
            if pd.notna(sample_data1[i][a]):
                temp.append(a)
            else:
                unknown_temp.append(a)
        sample_data_df = pd.DataFrame(sample_data1)
        dropped_1 = sample_data_df.iloc[:,[temp[0],temp[1]]].dropna()
        dropped_1_1 = dropped_1.drop(index=i)
        cor_1 = statistics.correlation(dropped_1.iloc[:,0], dropped_1.iloc[:,1])
        cor_1_1 = statistics.correlation(dropped_1_1.iloc[:,0], dropped_1_1.iloc[:,1])
        teacher_list.append(cor_1_1/cor_1)

        dropped2 = sample_data_df.iloc[:,[temp[0],temp[2]]].dropna()
        dropped2_2 = dropped2.drop(index=i)
        cor_2 = statistics.correlation(dropped2.iloc[:,0], dropped2.iloc[:,1])
        cor_2_2 = statistics.correlation(dropped2_2.iloc[:,0], dropped2_2.iloc[:,1])
        teacher_list.append(cor_2_2/cor_2)


        dropped3 = sample_data_df.iloc[:,[temp[1],temp[2]]].dropna()
        dropped3_3 = dropped3.drop(index=i)
        cor_3 = statistics.correlation(dropped3.iloc[:,0], dropped3.iloc[:,1])
        cor_3_3 = statistics.correlation(dropped3_3.iloc[:,0], dropped3_3.iloc[:,1])
        teacher_list.append(cor_3_3/cor_3)
        teacher_dic['one'] = (teacher_list[0] + teacher_list[1]) / 2
        teacher_dic['two'] = (teacher_list[0] + teacher_list[2]) / 2
        teacher_dic['three'] = (teacher_list[1] + teacher_list[2]) / 2

        #sorted_teachers = sorted(unknown_temp.items(), key=lambda x:x[1])
        for b in range(len(unknown_temp)):
            tempSum = 0
            avg_sum = 0
            total_len = 0
            total_z = []
            temp_array = np.array([])
            
            for j in range(len(temp)):
                        index = 0
                        weight = list(teacher_dic.values())[j]
                        dropped = sample_data_df.iloc[:,[temp[j],unknown_temp[b]]].dropna()
                        adjusted =  sample_data_df.iloc[i,temp[j]] - dropped[temp[j]].mean()
                        tempSum += dropped[unknown_temp[b]].mean()
                        #print(adjusted)
                        #mean_diff = inter_data[temp[j]][unknown_temp[b]]
                        corr = corr_table[temp[j]][unknown_temp[b]]

                        '''
                        z_score =  abs(adjusted) / statistics.stdev(dropped[temp[j]])
                        if z_score <= 1:
                            
                            weight = 0.8
                        elif z_score > 1 and z_score <= 2:
                            weight = 0.8
                        elif z_score > 2 and z_score <= 3:
                            
                            weight = 0.05
                        else:
                             
                            weight = 0.01 
                        '''
                        #common_length = len(dropped)
                        #total_len += common_length
                        #total_z.append(weight)
                        weighted = adjusted  *corr*weight
                        
                        temp_array = np.append(temp_array,[weighted,corr*weight])
            temp_array = temp_array.reshape((len(temp), 2))
            ''' 
            if max(total_z.values()) == 2:
              x = max(total_z, key = total_z.get)
              if x == 'one':
               selected = temp_array[temp_array[:, 2] == 1]
               not_selected = temp_array[temp_array[:, 2] != 1]
               not_selected[:, 0:2]*=0.4
               
               #avg_sum += not_sum_selected*0.4
               selected[:, 0:2]*=0.6
               final = np.append(selected,not_selected,axis=0)
               avg_sum = sum(final[:,0])/sum(final[:,1])
               
               #avg_sum += s
              elif x == 'two':
               selected = temp_array[temp_array[:, 2] == 2]
               not_selected = temp_array[temp_array[:, 2] != 2]
               not_selected[:, 0:2]*=0.4
               
               #avg_sum += not_sum_selected*0.4
               selected[:, 0:2]*=0.6
               final = np.append(selected,not_selected,axis=0)
               avg_sum = sum(final[:,0])/sum(final[:,1])
               
               
               #avg_sum += s
              elif x == 'three':
               selected = temp_array[temp_array[:, 2] == 3]
               not_selected = temp_array[temp_array[:, 2] != 3]
               not_selected[:, 0:2]*=0.4
               
               #avg_sum += not_sum_selected*0.4
               selected[:, 0:2]*=0.6
               final = np.append(selected,not_selected,axis=0)
               avg_sum = sum(final[:,0])/sum(final[:,1])
               
               #avg_sum += sum_selected*1.2
               #avg_sum = avg_sum/1.6

              elif x == 'four':
               selected = temp_array[temp_array[:, 2] == 4]
               not_selected = temp_array[temp_array[:, 2] != 4]
               not_selected[:, 0:2]*=0.4
               
               #avg_sum += not_sum_selected*0.4
               selected[:, 0:2]*=0.6
               final = np.append(selected,not_selected,axis=0)
               avg_sum = sum(final[:,0])/sum(final[:,1])
               
               #avg_sum += sum_selected*1.2
               #avg_sum = avg_sum/1.6
            else:
                avg_sum = np.sum(temp_array[:, 0])/np.sum(temp_array[:, 1])
            #average = sum(temp_array[:,0])/(sum(temp_array[:,1]))
            '''
            average = sum(temp_array[:,0])/(sum(temp_array[:,1]))
            #average = average + (tempSum/total_len)
            #average = round(average)
            #average = avg_sum
            average += tempSum/3
            average = round(average)
            if average >= 100:
                average = 100
            elif average <= 0:
                average = 0
            sample_data1[i][unknown_temp[b]] = average

    return sample_data1

def mixed(sample_data):
    
    
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
            sample_data_df = pd.DataFrame(sample_data1)
            tempSum = 0
            temp_array = np.array([])
            for j in range(len(temp)):
                        dropped = sample_data_df.iloc[:,[temp[j],unknown_temp[b]]].dropna()
                        adjusted =  sample_data_df.iloc[i,temp[j]] - dropped[temp[j]].mean()
                        #mean_diff = inter_data[temp[j]][unknown_temp[b]]
                        corr = corr_table[temp[j]][unknown_temp[b]]
                        weighted = adjusted*corr
                        tempSum += dropped[unknown_temp[b]].mean()
                        #tempSum += mean_diff
                        temp_array = np.append(temp_array,[weighted,corr])
                        
            temp_array = temp_array.reshape((len(temp), 2))          
            
            average = sum(temp_array[:,0])/sum(temp_array[:,1])
            average = average + (tempSum/3)
            average = round(average)
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
    # print(freq)
    commons = common_grade_table(pd.DataFrame(train_data))
    # print(commons)
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

    # print(test_dict)
    commons.to_csv("csv2.csv")
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
    truth = pd.read_csv("Implementation\Truth.csv")
    truth2 = truth.drop('Project ID', inplace=False, axis=1)
    truth2 = truth2.T.to_numpy().flatten()

    avg_p = []
    rmse = 0
    error = []
    # calculate the mean for all projects
    #print(len(predictions))
    for i in range(len(predictions)):
        avg_p.append(np.mean(predictions[i]))
    # Calculates RMSE
    for i in range(len(avg_p)):
        
        error.append(abs(avg_p[i] - truth2[i]))
        rmse += ((avg_p[i] - truth2[i]) ** 2)
    #print(error)
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
#fifth_optimization(10)
original_data = df2.to_numpy()
train2 = MLModel_VariationC(meanCorrelation(df2),df2.to_numpy())
#train2 = collaborative(df2.to_numpy())
#dff = pd.DataFrame(train2)
#dff.to_csv('mixed.csv')
#err2, rmse2 = validation_source_truth(train2)
#print("RMSE value: " + str(rmse2))
#print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
#print("Error average:" + str(np.mean(err2)))
#NN(df2.to_numpy(),meanCorrelation(df2))
#X = train2
   
'''
rank = 5
model = NMF(n_components=rank, init='random', random_state=0, max_iter = 1300)

# fit model to the matrix
model.fit(X)

# predict missing values
W = model.transform(X)
H = model.components_
X_pred = np.dot(W, H)
'''
from sklearn.impute import KNNImputer
M = df2.to_numpy()

# Perform SVD on the matrix


# Set the rank of the low-rank approximation
imputer = KNNImputer(n_neighbors=2)
A_imputed = imputer.fit_transform(M)

# Compute the SVD decomposition of the imputed matrix
U, s, Vt = np.linalg.svd(train2, full_matrices=False)

# Set the rank of the low-rank approximation
k = 2

# Compute the low-rank approximation of the imputed matrix
Ak = U[:, :k].dot(np.diag(s[:k])).dot(Vt[:k, :])

# Predict missing values using the low-rank approximation
A_predicted = np.where(np.isnan(M), Ak, train2)


# Print the original matrix, the imputed matrix, and the predicted matrix with missing values

#print("Imputed matrix:\n", A_imputed)
#print("Predicted matrix with missing values:\n", A_predicted)
rank = 5
model = NMF(n_components=rank, init='random', random_state=0, max_iter = 1300)

# fit model to the matrix
model.fit(A_imputed)
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

# Load your preprocessed data matrix here
data = A_predicted
y_other = pd.read_csv('C:/Users/Ege/Desktop/FairGrading-1/Implementation/Truth.csv')
y_other.drop('Project ID',inplace=True, axis = 1)
    
y_other = np.array(y_other)
#y_other = y_other.reshape(-1)
data = np.append(data,y_other , axis=1)
# Define the number of folds for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define your neural network architecture
model = Sequential()

model.add(Dense(32, input_dim=25, activation='relu'))
model.add(Dense( 64 , activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')


# Train and evaluate the neural network using cross-validation
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
np.random.shuffle(data)
X_train, X_test = data[:800,:-1], data[800:,:-1]
y_train, y_test = data[:800,-1], data[800:,-1]
model.fit(X_train, y_train, epochs=1000, verbose=0)
score2 = model.evaluate(X_train,y_train)
score = model.evaluate(X_test, y_test, verbose=0)
print('Fold mean squared error:', score)
print('Fold mean squared error:', score2)
predicted = np.delete(data, -1, 1)
predicted2 = model.predict(predicted)

#mask = np.isnan(original_data)
#original_data[mask] = predicted[mask]
#print(predicted)

# Calculate the average mean squared error across all folds


'''
'''
err2, rmse2 = validation_source_truth(predicted)
print("RMSE value: " + str(rmse2))
print("Max & Min : " + str(max(err2)) + " & " + str(min(err2)))
print("Error average:" + str(np.mean(err2)))
