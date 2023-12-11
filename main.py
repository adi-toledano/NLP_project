"""
 Chosen data sets are: freesolv, Lipophilicity, BBBP, BACE
 1) Lipophilicity and freesolv are regression problems, BBBP and BACE are classification problems
 2) The metric used for classification is AUC-ROC
 3) The metric used for regression is RMSE

 Requirements: need to open project in Conda environment and install the library 'deepchem'
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.molnet import load_bbbp
from deepchem.molnet import load_lipo
from deepchem.molnet import load_freesolv
from deepchem.molnet import load_bace_classification
import numpy as np
from sklearn import svm, metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


def extract_data(dataset):
    train_dataset, valid_dataset, test_dataset = dataset
    return train_dataset.ids, valid_dataset.ids, test_dataset.ids, train_dataset.y, valid_dataset.y, test_dataset.y


# Use CountVectorizer to vectorize based on characters and n-grams
def vectorize_fit_transform(train, N):
    combined_text = ' '.join(train)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, N), lowercase=False)
    vectorizer.fit_transform([combined_text])
    # Get the feature names (characters and n-grams)
    # feature_names = vectorizer.get_feature_names_out()
    return vectorizer


# Vectorize each word based on the features obtained from the training set vectorizer
def vectorize_transform(set, vectorizer):
    vectorized_words = []
    for word in set:
        word_vector = vectorizer.transform([word])
        vectorized_words.append(word_vector.toarray()[0])
    return vectorized_words


def load_and_preprocess_data(load_dataset_func, split):
    tasks, datasets, transformers = load_dataset_func(reload=False, split=split)
    x_train, x_val, x_test, y_train, y_val, y_test = extract_data(datasets)

    y_train = [row[0] for row in y_train]
    y_val = [row[0] for row in y_val]
    y_test = [row[0] for row in y_test]

    return x_train, x_val, x_test, y_train, y_val, y_test


def vectorize_data(x_train, x_val, x_test, N):
    vec = vectorize_fit_transform(x_train, N)
    x_train_vec = vectorize_transform(x_train, vec)
    x_val_vec = vectorize_transform(x_val, vec)
    x_test_vec = vectorize_transform(x_test, vec)
    return x_train_vec, x_val_vec, x_test_vec


def train_classifier(x_train_vec, y_train, x_val_vec, y_val, N):
    print(f"Running SVM on validation set with N={N}")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']}

    grid_search = GridSearchCV(svm.SVC(probability=True), param_grid=param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(x_train_vec, y_train)
    best_model = grid_search.best_estimator_
    val_pred = best_model.predict(x_val_vec)

    # Calculate different metrics
    val_accuracy = metrics.accuracy_score(y_val, val_pred)
    val_f1 = metrics.f1_score(y_val, val_pred)
    val_auc_roc = roc_auc_score(y_val, best_model.predict_proba(x_val_vec)[:, 1])

    print(f'{best_model} achieved accuracy of {val_accuracy}, f1 accuracy of {val_f1} and AUC-ROC of {val_auc_roc}')
    return best_model, val_auc_roc


def test_best_classifier(classifier_results, y_test):
    # Find the best N value to use for prediction based on best AUC-ROC score
    N, best_classifier = max(classifier_results.items(), key=lambda x: x[1][1])
    best_model, _, x_test_vec = best_classifier

    print(f'Running best classifier {best_model} on test set with N={N}')
    test_pred = best_model.predict(x_test_vec)

    # Calculate different metrics
    test_accuracy = metrics.accuracy_score(y_test, test_pred)
    test_f1 = metrics.f1_score(y_test, test_pred)
    test_auc_roc = roc_auc_score(y_test, best_model.predict_proba(x_test_vec)[:, 1])

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test F1-score: {test_f1}')
    print(f'Test AUC-ROC: {test_auc_roc}')


def train_regressor(x_train_vec, y_train, x_val_vec, y_val, N):
    print(f"Running KRR on validation set with N={N}")
    krr = KernelRidge()

    # Define hyperparameters and their ranges for grid search
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],  # Regularization parameter
        'kernel': ['rbf'],  # Choice of kernel functions
        'gamma': [0.01, 0.1, 1.0]  # Kernel coefficient for 'rbf' kernel
    }
    grid_search = GridSearchCV(estimator=krr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_vec, y_train)

    best_model = grid_search.best_estimator_
    val_pred = best_model.predict(x_val_vec)

    mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(mse)

    print(f'{best_model} achieved RMSE of {val_rmse}')
    return best_model, val_rmse


def test_best_regressor(regression_results, y_test):
    # Find the best N value to use for prediction based on best RMSE
    N, best_regressor = min(regression_results.items(), key=lambda x: x[1][1])
    best_model, _, x_test_vec = best_regressor

    print(f'Running best regressor {best_model} on test set with N={N}')
    test_pred = best_model.predict(x_test_vec)

    mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(mse)

    print(f'Test RMSE: {test_rmse}')


def learn_classification(load_dataset_func, split="random"):
    N_grams = [1, 2, 3]
    classifier_results = {}
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(load_dataset_func, split)

    # Tune choice of N
    for N in N_grams:
        x_train_vec, x_val_vec, x_test_vec = vectorize_data(x_train, x_val, x_test, N)
        best_model, val_auc_roc = train_classifier(x_train_vec, y_train, x_val_vec, y_val, N)
        classifier_results[N] = (best_model, val_auc_roc, x_test_vec)

    test_best_classifier(classifier_results, y_test)


def learn_regression(load_dataset_func, split="random"):
    N_grams = [1, 2, 3]
    regression_results = {}
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(load_dataset_func, split)

    # Tune choice of N
    for N in N_grams:
        x_train_vec, x_val_vec, x_test_vec = vectorize_data(x_train, x_val, x_test, N)
        best_model, val_rmse = train_regressor(x_train_vec, y_train, x_val_vec, y_val, N)
        regression_results[N] = (best_model, val_rmse, x_test_vec)

    test_best_regressor(regression_results, y_test)


def main():
    # Regression problems

    # Load FreeSolv dataset
    learn_regression(load_freesolv, split="random")

    # Load Lipophilicity dataset
    learn_regression(load_lipo, split="random")

    # Classification problems

    # Load BBBP dataset
    # learn_classification(load_bbbp, split="scaffold")

    # Load BACE dataset
    # learn_classification(load_bace_classification, split="scaffold")


if __name__ == "__main__":
    main()
