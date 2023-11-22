"""
 Chosen data sets are: BBBP, HIV, clinTox, BACE
 1) BBBP, HIV, BACE contain 1 task, clinTox contains 2 tasks.
 2) BBBP, BACE, clinTox contains less than 2100 samples, HIV contains around 40,000 (good for comparison)
 3) The metric used for evaluation is AUC-ROC

 Requirements: need to open project in Conda environment and install the library 'deepchem'
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from deepchem.molnet import load_bbbp
from deepchem.molnet import load_clintox
from deepchem.molnet import load_hiv
from deepchem.molnet import load_bace_classification
from sklearn import svm, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def extract_data_X(dataset):
    train_dataset, valid_dataset, test_dataset = dataset
    return train_dataset.ids, valid_dataset.ids, test_dataset.ids


def extract_data_y(dataset):
    train_dataset, valid_dataset, test_dataset = dataset
    return train_dataset.y, valid_dataset.y, test_dataset.y


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


def load_and_preprocess_data(load_dataset_func, task_num):
    tasks, datasets, transformers = load_dataset_func()
    x_train, x_val, x_test = extract_data_X(datasets)
    y_train, y_val, y_test = extract_data_y(datasets)

    y_train = [row[task_num] for row in y_train]
    y_val = [row[task_num] for row in y_val]
    y_test = [row[task_num] for row in y_test]

    return x_train, x_val, x_test, y_train, y_val, y_test


def vectorize_data(x_train, x_val, x_test, N):
    vec = vectorize_fit_transform(x_train, N)
    x_train_vec = vectorize_transform(x_train, vec)
    x_val_vec = vectorize_transform(x_val, vec)
    x_test_vec = vectorize_transform(x_test, vec)
    return x_train_vec, x_val_vec, x_test_vec


def train_classifier(x_train_vec, y_train, x_val_vec, y_val, N):
    print(f"Running SVM on validation set with N={N}")
    grid_search = GridSearchCV(svm.SVC(probability=True),
                               {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}, cv=3)
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
    # Find the best N value to use for prediction based on best f1 score
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


def learn(load_dataset_func, task_num=0):
    N_grams = [1, 2]
    classifier_results = {}
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(load_dataset_func, task_num)

    # Tune choice of N
    for N in N_grams:
        x_train_vec, x_val_vec, x_test_vec = vectorize_data(x_train, x_val, x_test, N)
        best_model, val_auc_roc = train_classifier(x_train_vec, y_train, x_val_vec, y_val, N)
        classifier_results[N] = (best_model, val_auc_roc, x_test_vec)

    test_best_classifier(classifier_results, y_test)


def main():
    # Load BBBP dataset
    learn(load_bbbp)

    # Load HIV dataset
    # This one takes lots of time because it is consisted from 40,000 samples
    # I added this large set for evaluation purposes but maybe it's too big, we'll see...
    # learn(load_hiv)

    # Load clintox dataset
    # clintox has 2 tasks, so need to classify them separately
    learn(load_clintox, task_num=0)
    learn(load_clintox, task_num=1)
    #
    # # Load BACE dataset
    learn(load_bace_classification)


if __name__ == "__main__":
    main()
