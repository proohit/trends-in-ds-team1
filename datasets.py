import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

wine_quality_red1 = pd.read_csv(
    './data/winequality-red-1.csv', sep=";", decimal=',')
wine_quality_red2 = pd.read_csv(
    './data/winequality-red-2.csv', sep=";", decimal=',')

wine_quality_white1 = pd.read_csv(
    './data/winequality-white-1.csv', sep=";", decimal=',')
wine_quality_white2 = pd.read_csv(
    './data/winequality-white-2.csv', sep=";", decimal=',')

wine_red = pd.merge(left=wine_quality_red1,
                    right=wine_quality_red2, on='ID', how='outer')
wine_white = pd.merge(left=wine_quality_white1,
                      right=wine_quality_white2, on='ID')


def wine_red_dataset():
    return pd.merge(left=wine_quality_red1, right=wine_quality_red2, on='ID')


def wine_white_dataset():
    return pd.merge(left=wine_quality_white1, right=wine_quality_white2, on='ID')


def get_wine_red_features():
    wine_red = wine_red_dataset()
    return wine_red.drop('ID', axis=1).drop('quality', axis=1)


def get_wine_red_label():
    wine_red = wine_red_dataset()
    return wine_red['quality']


def show_mvs(data):
    display(data[data.isnull().any(axis=1)])


def drop_mvs_all_inplace(data):
    data.dropna(how='all', inplace=True)


def plot_corr(data, feature1, title):
    return data.corrwith(data[feature1]).plot.bar(
        figsize=(15, 7.5), title=title, fontsize=15,
        rot=45, grid=True)


def get_logist_regression_score(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.7, random_state=1)
    lr = LogisticRegression().fit(x_train, y_train)
    return lr.score(x_test, y_test)


def normalize_feature(data, feature):
    scaler2 = preprocessing.MinMaxScaler()
    feature_normalized = scaler2.fit_transform(data[[feature]])
    new_data = data.copy()
    new_data[feature] = feature_normalized
    return new_data


def standardize_feature(data, feature):
    scaler2 = preprocessing.StandardScaler()
    feature_normalized = scaler2.fit_transform(data[[feature]])
    new_data = data.copy()
    new_data[feature] = feature_normalized
    return new_data
