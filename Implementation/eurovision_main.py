import time

import numpy as np
import pandas as pd
import matplotlib

seconds = time.time()
pd.options.display.max_rows = 999
pd.options.display.max_columns = 200
df = pd.read_csv("eurovision_song_contest_1975_2019.csv")
df.columns = ['year', 'final', 'edition', 'votetype', 'countryfrom', 'countryto', 'points', 'duplicate']


# max year included
# TODO in future maybe we can derive more efficient algorithm this runs for O(n^2*m) m: the number of years
#  n: number of countries participate selected year
def adjustPoints(data, minYear, maxYear):
    # data2 = data.copy(deep=False).query("`year`>= @minYear and `year` <= @maxYear")

    data.insert(6, "adjustedPoints", 0)
    for k in range(minYear, maxYear + 1):
        one_year = data.query("`year` == @k ")
        countries = one_year['countryfrom'].unique()

        length_countries = len(one_year['countryfrom'].unique())

        # for each countries find their related points, for related country calculate adj point and insert as a new
        # value
        # i is index of country who are voted, j is index of country who take a vote
        for j in range(length_countries):
            # Some countries cannot take vote because of elimination in the semi-finals
            if one_year[one_year['countryto'] == countries[j]].empty:
                continue
            for i in range(length_countries):
                # Do no take if countries are same
                if countries[i] == countries[j]:
                    continue
                avg = np.average(one_year[one_year['countryto'] == countries[j]].iloc[:, 5:6])
                country = one_year[one_year['countryto'] == countries[j]]
                point = (country[country['countryfrom'] == countries[i]])
                if point.empty:
                    continue
                # Get value from point variable
                index = point.index
                point = point.iloc[0, 5]

                diff = point - avg
                data.loc[index, 'adjustedPoints'] = diff

    return data


def basicClean(df, minYears=5, last_participation=8):
    df2 = df.copy()

    # Removing duplicates
    df2['duplicate'] = df2['duplicate'].apply(lambda x: True if x == 'x' or x is True else False)
    df2 = df2.query('duplicate == False').drop(columns=['duplicate'])

    # Stantardazing countries names.
    def applyRename(x):
        renamings = {
            'North Macedonia': 'Macedonia',
            'F.Y.R. Macedonia': 'Macedonia',
            'The Netherands': 'Netherlands',
            'The Netherlands': 'Netherlands',
            'Bosnia & Herzegovina': 'Bosnia',
        }
        return renamings[x] if x in renamings else x

    df2['countryfrom'] = df2['countryfrom'].apply(applyRename)
    df2['countryto'] = df2['countryto'].apply(applyRename)

    # replicating  Yugoslavia's results to countries that part of it.
    division = {
        'Yugoslavia': ['Macedonia', 'Serbia', 'Montenegro', 'Slovenia', 'Bosnia', 'Croatia'],
        'Serbia & Montenegro': ['Serbia', 'Montenegro'],
    }

    df2['countryfrom'] = df2['countryfrom'].apply(lambda x: division[x] if x in division else x)
    df2['countryto'] = df2['countryto'].apply(lambda x: division[x] if x in division else x)
    df2 = df2.explode('countryfrom').explode('countryto')

    # removing countries with less then 8 participations and not active in the last 5 years
    toKeep = df2.groupby('countryfrom').apply(lambda x: pd.Series({
        'years': x['year'].nunique(),
        'last_participation': df2['year'].max() - x['year'].max(),
    })).query(f'years >= {minYears} and last_participation <= {last_participation}').reset_index()['countryfrom']

    # display(HTML("<p>ignored countries: %s</p>" % ', '.join(
    #     df2[df2['countryfrom'].isin(toKeep) == False]['countryfrom'].unique())))

    df2 = df2[df2['countryfrom'].isin(toKeep)]
    df2 = df2[df2['countryto'].isin(toKeep)]

    # keep only the points received at the highest stage (finals/semifinals)
    df2['finalcode'] = df2.final.map({'f': 1, 'sf': 2, 'sf1': 2, 'sf2': 2})
    temp1 = df2.groupby(['countryto', 'year']).agg({'finalcode': 'min'})
    df2 = pd.merge(df2, temp1, on=['countryto', 'year', 'finalcode'], how='inner')

    assert len(df2.groupby(['countryfrom', 'countryto', 'year']).agg({'final': 'nunique'}).query('final >1')) == 0

    df2.drop(columns=['finalcode', 'edition'], inplace=True)

    return df2.reindex()


# max year included
def split(data, minYear, maxYear, c_no):
    data2 = data.copy().query("`year`>= @minYear and `year` <= @maxYear")
    test = pd.DataFrame()

    for i in range(minYear, maxYear + 1):
        one_year = data2.query("`year` == @i ")

        countries = one_year['countryto'].unique()
        countries2 = countries.copy()

        length_countries = len(countries)
        for k in range(length_countries):
            # competitors also juries
            selected_competitor = countries[k]
            df_competitor = one_year[one_year['countryto'] == selected_competitor]
            countries2 = df_competitor['countryfrom'].unique()

            countries2 = np.delete(countries2, np.argwhere(countries2 == selected_competitor))
            a = len(countries2)

            for j in range(a - c_no):
                r = np.random.randint(a - j)
                # partition the data frame
                country = countries2[r]
                x = one_year[one_year['countryfrom'] == country]

                test = pd.concat([test, x[x['countryto'] == selected_competitor]])

                countries2 = np.delete(countries2, r)
    data = data.drop(test.index)

    return data, test


# max year included
def prediction(data, test, minYear, maxYear):
    data = adjustPoints(data, minYear, maxYear)

    average_table = pd.DataFrame()
    std_table = pd.DataFrame()
    mse = 0

    for i in range(minYear, maxYear + 1):
        one_year = data.query('`year` == @i')

        for row in one_year.itertuples():
            if row[4] in average_table.index:
                if row[5] in average_table.columns:
                    if average_table.at[row[4], row[5]] == 0:
                        average_table.at[row[4], row[5]] = [row[7], 1]
                    else:
                        df_list = average_table.at[row[4], row[5]]
                        df_list[0] += row[7]
                        df_list[1] += 1
                else:
                    average_table.insert(len(average_table.columns), row[5], 0)
                    average_table = average_table.astype('object')
                    average_table.at[row[4], row[5]] = [row[7], 1]
            else:
                if row[5] in average_table.columns:

                    arr = [0 for a in range(len(average_table.columns))]
                    average_table.loc[row[4]] = arr
                    average_table = average_table.astype('object')
                    average_table.at[row[4], row[5]] = [row[7], 1]
                else:
                    average_table.insert(len(average_table.columns), row[5], 0)
                    arr = [0 for a in range(len(average_table.columns))]
                    average_table.loc[row[4]] = arr
                    average_table = average_table.astype('object')
                    average_table.at[row[4], row[5]] = [row[7], 1]

    for i in range(minYear, maxYear + 1):
        one_year = data.query('`year` == @i')
        adjusted_average = 0
        for row in one_year.itertuples():

            adjusted_average = average_table.at[row[4], row[5]][0] / average_table.at[row[4], row[5]][1]

            std = (row[7] - adjusted_average) ** 2
            if row[4] in std_table.index:
                if row[5] in std_table.columns:
                    std_table.at[row[4], row[5]] += std
                else:
                    std_table.insert(len(std_table.columns), row[5], 0)
                    std_table.at[row[4], row[5]] = std
            else:
                if row[5] in std_table.columns:
                    arr = [0 for i in range(len(std_table.columns))]
                    std_table.loc[row[4]] = arr
                    std_table.at[row[4], row[5]] = std
                else:
                    std_table.insert(len(std_table.columns), row[5], 0)
                    arr = [0 for i in range(len(std_table.columns))]
                    std_table.loc[row[4]] = arr
                    std_table.at[row[4], row[5]] = std

    for i in average_table.index:
        for j in average_table.columns:
            if average_table.at[i, j] != 0:
                std = np.sqrt(std_table.at[i, j] / average_table.at[i, j][1])
                std_table.at[i, j] = std

    for test_row in test.itertuples():
        year = test_row[1]
        countryto = test_row[5]

        x = data.query('`year` == @year and `countryto` == @countryto')
        point = x['points'].mean()
        bias = 0
        if test_row[4] in average_table.index and test_row[5] in average_table.columns:
            if average_table.at[test_row[4], test_row[5]] != 0:
                bias = average_table.at[test_row[4], test_row[5]][0] / average_table.at[test_row[4], test_row[5]][1]

        p_value = point + bias

        mse += (test_row[6] - p_value) ** 2

    mse = mse / len(test)
    rmse = np.sqrt(mse)
    nu_rmse = 0
    count = 0
    # prediction of average performance of country in the year
    years = test['year'].unique()
    for y in years:
        one_year = test[test['year'] == y]
        countries = one_year['countryto'].unique()
        for j in range(len(countries)):

            l = len(test[test['countryto'] == countries[j]].iloc[:, 5:6])
            l += len(data.query(' `year` == @y and `countryto` == @countries[@j] ').iloc[:, 5:6])

            s = np.sum(test[test['countryto'] == countries[j]].iloc[:, 5:6].to_numpy())
            s += np.sum(data.query(' `year` == @y and `countryto` == @countries[@j] ').iloc[:, 5:6].to_numpy())
            average = s / l

            countries2 = data.query(' `year` == @y and `countryto` == @countries[@j] ').loc[:,
                         'countryfrom'].to_numpy()
            avg_hat = 0
            count2 = 0
            for i in range(len(countries2)):
                row = data.query(
                    ' `year` == @y and `countryto` == @countries[@j] and `countryfrom` == @countries2[@i] ')
                perf = data.query(' `year` == @y and `countryto` == @countries[@j] ')
                perf = perf['points'].mean()  # this is country performance of the year
                p = row.iloc[0, 5]
                bias = 0
                if countries2[i] in average_table.index and countries[j] in average_table.columns:
                    if average_table.at[countries2[i], countries[j]] != 0:
                        bias = average_table.at[countries2[i], countries[j]][0] / \
                               average_table.at[countries2[i], countries[j]][1]

                        # this is current adjusted value of the country j
                        if std_table.at[countries2[i], countries[j]] != 0:
                            standard_deviation = std_table.at[countries2[i], countries[j]]
                            coefficient = 1 / (standard_deviation + 0.5)
                            avg_hat += (p + bias) * coefficient
                            count2 += coefficient
                        else:
                            avg_hat += p + bias
                            count2 += 1
                else:
                    avg_hat += p + bias
                    count2 += 1
            avg_hat = avg_hat / count2

            nu_rmse += (average - avg_hat) ** 2
            count += 1
    nu_rmse = np.sqrt(nu_rmse / count)
    return mse, rmse, nu_rmse


def validation_for_randomized(data, minYear, maxYear, iteration_no):
    mse_avg = 0
    rmse_avg = 0
    nu_rmse_avg = 0
    for i in range(iteration_no):
        train_data, test_data = split(data, 2019, 2019, 4)
        mse, r, nu = prediction(train_data, test_data, minYear, maxYear)
        mse_avg += mse
        rmse_avg += r
        nu_rmse_avg += nu

    mse_avg = mse_avg / iteration_no
    rmse_avg = rmse_avg / iteration_no
    nu_rmse_avg /= iteration_no
    return mse_avg, rmse_avg, nu_rmse_avg


df2 = basicClean(df)
df3 = df2.query(' `final` == "f" ')

mse, rmse, nu_rmse = validation_for_randomized(df3, 1975, 2018, 20)
print(mse, rmse)
print("RMSE for predicting performance of a country : ", nu_rmse)
s = time.time()
print(seconds - s)
