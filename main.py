import pandas as pd
import numpy as np
import openpyxl
from preprocessing import preprocess, preprocess_prediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# Load data
data_read = pd.read_csv("data/dataCase1.csv")
data = data_read.head(100)
X_predictions_src = data_read.tail(1000)

Y = data.iloc[:, 0]
X = data.iloc[:, 1:]
X_train_src, X_test_src, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train_src.reset_index(drop=True, inplace=True)
X_test_src.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

preprocessing_params = {"No preprocessing":    {"high_feature_correlation": 0.9999,
                                                "low_output_correlation":   0.00000001,
                                                "percentage_variance_explained_PCA": 1,
                                                "kernel_pca_components": 0,
                                                "kernel_for_pca": "sigmoid"},
                        "Light preprocessing": {"high_feature_correlation": 0.9,
                                                "low_output_correlation":   0.02,
                                                "percentage_variance_explained_PCA": 1,
                                                "kernel_pca_components": 0,
                                                "kernel_for_pca": "sigmoid"},
                        "Heavy preprocessing": {"high_feature_correlation": 0.9999,
                                                "low_output_correlation":   0.00000001,
                                                "percentage_variance_explained_PCA": 0.99,
                                                "kernel_pca_components": 0,
                                                "kernel_for_pca": "sigmoid"}}
model_names = ["elastic", "forest", "svm"]


results_training = pd.DataFrame([])
results_test = pd.DataFrame([])
best_models = {}
y_hat = []
y_hat_predict = []

for name, param in preprocessing_params.items():
    X_train, delete_features, pca, kernel_pca = preprocess(X_train_src, y_train,
                                               high_feature_correlation=param["high_feature_correlation"],
                                               low_output_correlation=param["low_output_correlation"],
                                               percentage_variance_explained_PCA=param["percentage_variance_explained_PCA"],
                                               kernel_pca_components=param["kernel_pca_components"],
                                               kernel_for_pca=param["kernel_for_pca"])

    X_test = preprocess_prediction(X_test_src, delete_features, pca, kernel_pca)
    X_predictions = preprocess_prediction(X_predictions_src.iloc[:, 1:], delete_features, pca, kernel_pca)


    alphas_size = 20
    min_alpha = 0.1
    max_alpha = 1.5
    l1_ratio_size = 10
    min_l1_ratio = 0.01
    max_l1_ratio = 0.99
    gs_elastic_net = GridSearchCV(estimator=ElasticNet(fit_intercept=True),
                                  param_grid={'alpha':    np.linspace(min_alpha, max_alpha, alphas_size).tolist(),
                                              'l1_ratio': np.linspace(min_l1_ratio, max_l1_ratio, l1_ratio_size).tolist()},
                                  scoring='neg_mean_squared_error',
                                  cv=16, iid=False, return_train_score=True)

    gs_elastic_net.fit(X_train, y_train)
    best_elastic = gs_elastic_net.best_estimator_
    validation_mse_elastic = gs_elastic_net.best_score_
    elastic_pd_results = pd.DataFrame(gs_elastic_net.cv_results_)
    training_mse_elastic = elastic_pd_results.loc[elastic_pd_results['rank_test_score'] == 1, ['mean_train_score']]

    # Random forest
    gs_forest = GridSearchCV(estimator=RandomForestRegressor(bootstrap=True),
                             param_grid={'max_depth': [10, 30, 50], 'n_estimators': [200, 400, 600]},
                             scoring='neg_mean_squared_error',
                             cv=16, iid=False, return_train_score=True)

    gs_forest.fit(X_train, y_train)
    best_forest = gs_forest.best_estimator_
    validation_mse_forest = gs_forest.best_score_
    forest_pd_results = pd.DataFrame(gs_forest.cv_results_)
    training_mse_forest = forest_pd_results.loc[forest_pd_results['rank_test_score'] == 1, ['mean_train_score']]

    # Sigmoid SVR
    gs_svm = GridSearchCV(estimator=svm.SVR(kernel='sigmoid', coef0=0, gamma="auto"),
                          param_grid={'C': [2.0, 4.0, 6.0, 8.0, 10.0]},
                          scoring='neg_mean_squared_error',
                          cv=16, iid=False, return_train_score=True)

    gs_svm.fit(X_train, y_train)
    best_svm = gs_svm.best_estimator_
    validation_mse_svm = gs_svm.best_score_
    svm_pd_results = pd.DataFrame(gs_svm.cv_results_)
    training_mse_svm = svm_pd_results.loc[svm_pd_results['rank_test_score'] == 1, ['mean_train_score']]

    results_training = results_training.append(pd.DataFrame({"training_mse_elastic": training_mse_elastic.iloc[0]["mean_train_score"],
                                           "validation_mse_elastic": validation_mse_elastic,
                                           "training_mse_forest": training_mse_forest.iloc[0]["mean_train_score"],
                                           "validation_mse_forest": validation_mse_forest,
                                           "training_mse_svm": training_mse_svm.iloc[0]["mean_train_score"],
                                           "validation_mse_svm": validation_mse_svm},
                                            index=[name]))

    models = [best_elastic, best_forest, best_svm]
    best_models[name] = models

    y_hat_elastic = best_elastic.predict(X_test)
    y_hat_forest = best_forest.predict(X_test)
    y_hat_svm = best_svm.predict(X_test)
    y_hat.append(y_hat_elastic)
    y_hat.append(y_hat_forest)
    y_hat.append(y_hat_svm)

    y_hat_predict_elastic = best_elastic.predict(X_predictions)
    y_hat_predict_forest = best_forest.predict(X_predictions)
    y_hat_predict_svm = best_svm.predict(X_predictions)
    y_hat_predict.append(y_hat_predict_elastic)
    y_hat_predict.append(y_hat_predict_forest)
    y_hat_predict.append(y_hat_predict_svm)

    results_test = results_test.append(pd.DataFrame({"elastic": mean_squared_error(y_test, y_hat_elastic),
                                                     "forest": mean_squared_error(y_test, y_hat_forest),
                                                     "svm": mean_squared_error(y_test, y_hat_svm)},
                                                     index=[name]))


y_hat = pd.DataFrame(np.array(y_hat))
y_hat_predict = pd.DataFrame(np.array(y_hat_predict))

y = pd.DataFrame(np.array(y_test))
writer = pd.ExcelWriter('results.xlsx')
results_training.to_excel(writer, 'results_training')
results_test.to_excel(writer, 'results_test')
y_hat.to_excel(writer, 'y_hat')
y.to_excel(writer, 'y')
y_hat_predict.to_excel(writer, 'y_hat_predict')
writer.save()


for name, key in best_models.items():
    for i, item in enumerate(key):
        joblib.dump(item, name+str(i)+".joblib.pkl", compress=9)

a=1