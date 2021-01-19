import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

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


def get_popularity_features(dataset=popularity_dataset()):
    return dataset.drop(' shares', axis=1).drop('days_article_dataset', axis=1)


def get_popularity_label(dataset=popularity_dataset()):
    return dataset.loc[:, [' shares']]


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


def get_mvs_of_feature(data, feature):
    if isinstance(feature, list):
        return data[data[feature].isna().any(axis=1) == True]
    else:
        return data[data[feature].isna()]


def get_mvs_of_features(data, features):
    return data[data[features].isna().any(axis=1) == True]


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


def get_cutpoints_for_percent(data, min, max, class_variable):
    total = data.shape[0]
    min_data = round(total - total * (1-min))
    max_data = round(total - total * (1-max))
    min_shares = data.iloc[min_data][class_variable]
    max_shares = data.iloc[max_data - 1][class_variable]
    return (min_shares, max_shares)


def get_class_for_value(value, cutpoints, classes):
    if value <= cutpoints[0][1]:
        return classes[0]
    elif value <= cutpoints[1][1]:
        return classes[1]
    elif value <= cutpoints[2][1]:
        return classes[2]
    elif value <= cutpoints[3][1]:
        return classes[3]
    elif value >= cutpoints[4][0]:
        return classes[4]


def get_noises(x, y, threshold=1):
    # Define 3 classifiers
    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()

    classname = y.name

    # merge 3 classifiers into one voting classifier
    eclf1 = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

    # train voting classifier with k-fold method
    y_pred = cross_val_predict(eclf1, x, y, cv=3)

    # save predictions, original quality and correct prediction boolean in data frame
    result = pd.DataFrame(y_pred, columns=['Prediction'], index=x.index)
    result[classname] = y
    result['Correct Prediction'] = abs(
        result['Prediction'] - result[classname]) < threshold

    # select all incorrect predicted data
    print('False predictions')
    delta_result = result[result['Correct Prediction'] == False]
    return (result, delta_result)


def get_outliers(x, y, contamination=0.1):
    iso = IsolationForest(contamination=contamination)
    y_out = iso.fit_predict(x)

    # build a mask to select all rows that are not outliers (inlier=1, outlier=-1)
    mask = y_out != 1
    X_outliers, y_outliers = x[mask], y[mask]

    # Inliers vs. Outliers
    print("Inliers: ", x.shape[0]-X_outliers.shape[0],
          "Outliers:", X_outliers.shape[0])

    # display(X_red)
    outliers = pd.DataFrame(
        X_outliers, columns=x.columns, index=X_outliers.index)
    outliers[y.name] = y_outliers
    return outliers


def accuracy_score_model(classifier, x, y):
    scores = cross_val_score(classifier, x, y, cv=5)
    accuracy = scores.mean()
    return accuracy


def calculate_scores(x, y):
    svcClf = SVC()
    svcScore = accuracy_score_model(svcClf, x, y)
    print("Accuracy SVM: %0.2f" % svcScore)
    logitClf = LogisticRegression()
    logitScore = accuracy_score_model(logitClf, x, y)
    print("Accuracy Logistic Regression: %0.2f" % logitScore)
    knnClf = KNeighborsClassifier()
    knnScore = accuracy_score_model(knnClf, x, y)
    print("Accuracy KNN: %0.2f" % knnScore)
    nbClf = GaussianNB()
    nbScore = accuracy_score_model(nbClf, x, y)
    print("Accuracy Naive Bayes: %0.2f" % nbScore)
    treeClf = DecisionTreeClassifier()
    treeScore = accuracy_score_model(treeClf, x, y)
    print("Accuracy Decision Tree: %0.2f" % treeScore)
    annClf = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=1)
    annScore = accuracy_score_model(annClf, x, y)
    print("Accuracy ANN: %0.2f" % annScore)


def get_sample_data(x, y, extra_column, size=.25):
    skf = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=1)
    for train_index, test_index in skf.split(x, y):
        sample_data = pd.DataFrame(x.iloc[train_index], columns=x.columns)
        sample_data[y.name] = y.iloc[train_index]
        sample_data[extra_column.name] = extra_column.iloc[train_index]
        return sample_data


def perform_forward_selection(data, label, columns, clf=LogisticRegression()):
    score = 0
    columns_to_use = []
    for column in columns:
        columns_to_use.append(column)
        x = data[columns_to_use]
        y = data[label]
        score_temp = accuracy_score_model(clf, x, y)
        if score_temp - score < 0:
            columns_to_use.remove(column)
            print(f"skipping feature {column}")
            continue
        score = score_temp
        print(f"Score with features {columns_to_use}: {score}")
    return columns_to_use
