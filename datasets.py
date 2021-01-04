import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score


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

popularity_data = pd.read_csv(
    './data/OnlineNewsPopularity.csv', sep=",", decimal=".")


def wine_red_dataset():
    return pd.merge(left=wine_quality_red1, right=wine_quality_red2, on='ID')


def wine_white_dataset():
    return pd.merge(left=wine_quality_white1, right=wine_quality_white2, on='ID')


def popularity_dataset():
    return popularity_data.drop('url', axis=1)


def get_wine_red_features():
    wine_red = wine_red_dataset()
    return wine_red.drop('ID', axis=1).drop('quality', axis=1)


def get_wine_white_label():
    wine_white = wine_white_dataset()
    return wine_white['quality']


def get_wine_white_features():
    wine_white = wine_white_dataset()
    return wine_white.drop('ID', axis=1).drop('quality', axis=1)


def get_wine_red_label():
    wine_red = wine_red_dataset()
    return wine_red['quality']


def show_mvs(data):
    mvs = data[data.isnull().any(axis=1)]
    display(mvs)
    return mvs


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


def get_logist_regression_kfold_score(x, y):
    lr = LogisticRegression()
    y_pred = cross_val_predict(lr, x, y)
    score = accuracy_score(y, y_pred)
    return score


def get_kfold_score(clf, x, y):
    y_pred = cross_val_predict(clf, x, y)
    score = accuracy_score(y, y_pred)
    return score


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


def get_delta_mean_median(data):
    median = data.median()
    mean = data.mean()
    delta_mean_median = (median - mean).round(5)
    quotient = (delta_mean_median / mean) * 100
    quotient = np.sqrt(quotient ** 2)
    return (quotient, median)


def show_scatterplot(data, x, y, x_label='feature', y_label='label'):
    plt.scatter(y=data[y], x=data[x])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_kbest_sorted(selector, x, y, iterations):
    flt = SelectKBest(selector).fit(x, y)
    scores = flt.scores_
    best_features = pd.DataFrame(scores, index=x.columns, columns=['score'])
    for i in range(1, iterations):
        np.random.seed(i)
        flt = SelectKBest(selector).fit(x, y)
        scores = flt.scores_
        df = pd.DataFrame(scores, index=x.columns, columns=['score'])
        best_features = df + best_features
    best_features = best_features / i
    best_features = best_features.sort_values(by=['score'], ascending=False)
    return best_features


def get_vifs(x):
    vif_factors = pd.Series([variance_inflation_factor(x.values, i)
                             for i in range(x.shape[1])], index=x.columns)
    vif_factors = round(vif_factors, 2)
    return vif_factors
