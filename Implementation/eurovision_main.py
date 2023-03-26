import numpy as np
import pandas as pd
import matplotlib

pd.options.display.max_rows = 999
pd.options.display.max_columns = 200
df = pd.read_csv("eurovision_song_contest_1975_2019.csv")
df.columns = ['year', 'final', 'edition', 'votetype', 'countryfrom', 'countryto', 'points', 'duplicate']


# max year included
# TODO in future maybe we can derive more efficient algorithm this runs for O(n^2*m) m: the number of years
#  n: number of countries participate selected year
def adjustPoints(data, minYear, maxYear):
    data2 = data.copy().query("`year`>= @minYear and `year` <= @maxYear")

    data2.insert(6, "adjustedPoints", 0)
    for k in range(minYear, maxYear + 1):
        one_year = data2.query("`year` == @k ")
        countries = one_year['countryfrom'].unique()

        length_countries = len(one_year['countryfrom'].unique())

        # for each countries find their related points, for related country calculate adj point and insert as a new
        # value
        for j in range(length_countries):
            avg = np.average(one_year[one_year['countryto'] == countries[j]].iloc[:, 5:6])

            for i in range(length_countries):
                country = one_year[one_year['countryto'] == countries[j]]
                point = (country[country['countryfrom'] == countries[i]])
                if point.empty:
                    continue
                # Get value from point variable
                index = point.index
                point = point.iloc[0, 5]

                diff = point - avg
                data2.loc[index, 'adjustedPoints'] = diff

    return data2


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

    # removing countries with less then 5 participations and not active in the last 5 years
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
def prediction(data, test, minYear, maxYear, pMinYear, pMaxYear):
    data = adjustPoints(data, minYear, maxYear)

    average_table = pd.DataFrame()

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

                    arr = [[0] * 2] * len(average_table.columns)
                    average_table.loc[row[4]] = arr
                    average_table.at[row[4], row[5]] = [row[7], 1]
                else:
                    average_table.insert(len(average_table.columns), row[5], 0)
                    arr = [[0] * 2] * len(average_table.columns)
                    average_table.loc[row[4]] = arr
                    average_table.at[row[4], row[5]] = [row[7], 1]

    for test_row in test.itertuples():
        year = test_row[0]
        countryto = test_row[5]

        x = data.query('`year` == @year and `countryto` == @countryto')
        for data_row in x.itertuples():
            p_value = data_row[6] + data_row[]
    return 0


df2 = basicClean(df)
train_data, test_data = split(df2, 1986, 1986, 3)
avg_table = prediction(train_data, test_data, 1975, 1985, 1986, 1986)
