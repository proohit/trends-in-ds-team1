import pandas as pd

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
