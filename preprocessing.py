import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['backend'] = "Qt4Agg"
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA, KernelPCA


def print_misssing(X):
    print('Featrues missing data:')
    print(X.isna().sum(axis=0).sort_values(ascending=False))
    print('Observations missing data: ')
    print(X.isna().sum(axis=1).sort_values(ascending=False))
    print('Total missing values:')
    print(X.isna().sum(axis=1).sum(axis=0))


# Change k_last_columns of categorical data to one-out-of-k encoding
def categorical_encoding(x, k_last_columns):
    # Categorical data to one-out-of-k
    categorical = x.iloc[:, -k_last_columns:]
    # Fill missing values with dominant value in each column
    categorical = categorical.fillna(categorical.mode().iloc[0])
    values = np.array(categorical)

    categorical_onehot = np.zeros((values.shape[0], 21))
    idx = 0
    # integer encode
    for val in values.T:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(val)
        # print(integer_encoded)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        categorical_onehot[:, idx:idx + onehot_encoded.shape[1]] = onehot_encoded
        idx = idx + onehot_encoded.shape[1]

    categorical_onehot_pd = pd.DataFrame(categorical_onehot)

    return categorical_onehot_pd


def normalize(X, categorical_data):
    # Fill nan with mean of columns
    X = X.iloc[:, :-5]
    X = X.fillna(X.mean())
    # Concatenate continuous and categorical data
    X.reset_index(drop=True, inplace=True)
    categorical_data.reset_index(drop=True, inplace=True)
    X = pd.concat([X, categorical_data], axis=1)
    # Normalization
    X = (X - X.mean()) / X.std()
    X = X.fillna(0)

    return X


def pca_function(X, percentage_variance_explained_PCA = 1):

    pca = PCA(n_components=percentage_variance_explained_PCA)
    X_transformed = pca.fit_transform(X)

    eigenvalues = pca.explained_variance_
    #print("Eigenvalues from PCA:\n\n")
    #print(eigenvalues)
    print("PCA reduced %d features.\n\n" % (X.shape[1] - X_transformed.shape[1]))

    return X_transformed, pca


def preprocess(X, Y, PLOT_COV = False, PLOT_COV_CAT = False,
                  PRINT_MISSING_DATA = True, SIMPLIFY_FEATURES=True,
                  percentage_variance_explained_PCA=1,
                  high_feature_correlation = 0.8,
                  low_output_correlation = 0.01,
                  kernel_pca_components=0,
                  kernel_for_pca="linear"):

    if PRINT_MISSING_DATA:
        print_misssing(X)

    categorical_data = categorical_encoding(X, 5)

    if PLOT_COV_CAT:
        plt.interactive(True)
        plt.matshow(categorical_data.corr())
        plt.ioff()
        plt.show()
        plt.interactive(False)

    X = normalize(X, categorical_data)

    corr_X = X.corr()
    if PLOT_COV:
        plt.interactive(True)
        plt.matshow(corr_X)
        plt.ioff()
        plt.show()
        plt.interactive(False)

    corr = pd.concat([Y, X], axis=1).corr().values
    corr_X = corr[1:, 1:]
    corr_Y = corr[1:, 0]

    if SIMPLIFY_FEATURES:
        n_high_corr_X = ((corr_X>high_feature_correlation).sum() - corr_X.shape[0])/2
        print("Number of features with correlation of %d or more: %d\n\n" % (high_feature_correlation, n_high_corr_X))
        n_low_corr_Y = ((low_output_correlation > corr_Y) & (corr_Y > -low_output_correlation))
        print("Number of features with %d output correlation: %d\n\n" % (low_output_correlation, n_low_corr_Y.sum()))

        corr_X_upper_tri = np.triu(corr_X, 1)
        delete_features_corr_X = (corr_X_upper_tri>high_feature_correlation).sum(axis=0) != 0
        delete_features = np.logical_or(n_low_corr_Y, delete_features_corr_X)

        for i in range(X.shape[1]-len(delete_features)):
            delete_features = np.append(delete_features, False)

        X_new = np.array(X)[:, np.invert(delete_features)]

        print("%d features were deleted." % (X.shape[1]-X_new.shape[1]))

        X = X_new

    pca = 0
    if percentage_variance_explained_PCA != 1:
        X, pca = pca_function(X, percentage_variance_explained_PCA)

    kernel_pca = 0
    if kernel_pca_components != 0:
        kernel_pca = KernelPCA(n_components=kernel_pca_components, kernel=kernel_for_pca)
        X = kernel_pca.fit_transform(X)

    return X, delete_features, pca, kernel_pca


def preprocess_prediction(X, delete_features, pca, kernel_pca, SIMPLIFY_FEATURES = True):

    categorical_data = categorical_encoding(X, 5)
    X = normalize(X, categorical_data)

    if SIMPLIFY_FEATURES:
        X = np.array(X)[:, np.invert(delete_features)]
        if pca != 0:
            X = pca.transform(X)
        if kernel_pca != 0:
            X = kernel_pca.fit_transform(X)

    return X
