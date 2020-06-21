##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from util import util
from util.VisualizeDataset import VisualizeDataset

pd.set_option('display.max_rows',(None))

user = 1600
# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('../intermediate_datafiles/')
DATASET_FNAME = f'chapter5_{user}_result_ws30.csv'
RESULT_FNAME = f'chapter7_classification_{user}_result.csv'
EXPORT_TREE_PATH = Path('../figures/crowdsignals_ch7_classification/')

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 10

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us create our visualization class again.

# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Select subsets of the features that we will consider:

basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','gyr_watch_x','gyr_watch_y','gyr_watch_z']
pca_features = ['pca_1','pca_2']
time_features = [name for name in dataset.columns if '_temp_' in name]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print('#basic features: ', len(basic_features))
print('#PCA features: ', len(pca_features))
print('#time features: ', len(time_features))
print('#frequency features: ', len(freq_features))
cluster_features = ['cluster']
print('#cluster features: ', len(cluster_features))
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

selected_features = ['acc_phone_z_temp_mean_ws_120', 'gyr_phone_x_temp_std_ws_120', 'acc_phone_z',
                     'acc_phone_x_temp_std_ws_120', 'acc_watch_y_temp_mean_ws_120', 'pca_2_temp_mean_ws_120', 'pca_1_temp_mean_ws_120']
learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']
N_KCV_REPEATS = 5

scores_over_all_algs = []


datasets = []
print("Reading in datasets")
for users in range(1600, 1611):
    DATA_PATH = Path('../intermediate_datafiles/')
    DATASET_FNAME = f'chapter5_{users}_result_ws30.csv'
    RESULT_FNAME = f'chapter7_classification_{users}_result.csv'
    EXPORT_TREE_PATH = Path('../figures/crowdsignals_ch7_classification/')
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    dataset.index = pd.to_datetime(dataset.index)
    datasets.append(dataset)
# for users in range(1621, 1631):
#     DATA_PATH = Path('./done/')
#     DATASET_FNAME = f'dataset_{users}_final.csv'
#     RESULT_FNAME = f'chapter7_classification_{users}_result.csv'
#     EXPORT_TREE_PATH = Path('../figures/crowdsignals_ch7_classification/')
#     try:
#         dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
#     except IOError as e:
#         print('File not found, try to run previous crowdsignals scripts first!')
#         raise e
#
#     dataset.index = pd.to_datetime(dataset.index)
#     datasets.append(dataset)

indices = prepare.split_multiple_datasets_classification(datasets=datasets, unknown_users=True,
                                                                                  class_labels=['label'], matching='like',
                                                                                  filter=True, training_frac=0.7,
                                                                                  temporal=True, for_cv=True)


print(indices)
print("Splitting cv datasets")
X_train_cv, X_test_cv, y_train_cv, y_test_cv = prepare.split_multiple_datasets_classification(datasets=[datasets[i] for i in indices], unknown_users=True,
                                                                                  class_labels=['label'], matching='like',
                                                                                  filter=True, training_frac=0.7,
                                                                                  temporal=True, fillna=True)

print("Performing cross-validation")

best_params, cv_train_y, cv_test_y, cv_train_prob_y, cv_test_prob_y = learner.random_forest(
    X_train_cv[features_after_chapter_5], y_train_cv, X_test_cv[features_after_chapter_5],
    gridsearch=True, print_model_details=False)

print("Splitting real datasets")
X_train, X_test, y_train, y_test = prepare.split_multiple_datasets_classification(datasets=datasets, unknown_users=True,
                                                                                  class_labels=['label'], matching='like',
                                                                                  filter=True, training_frac=0.7,
                                                                                  temporal=True)

_, class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
    X_train[features_after_chapter_5], y_train, X_test[features_after_chapter_5],
    gridsearch=False, print_model_details=True, criterion=best_params['criterion'], min_samples_leaf=best_params['min_samples_leaf'],
    n_estimators=best_params['n_estimators'])

test_cm = eval.confusion_matrix(y_test, class_test_y, class_train_prob_y.columns)

DataViz = VisualizeDataset(user, __file__, skip_plots=2)

DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)

print(f'Accuracy: {eval.accuracy(y_test, class_test_y)}')
print(f'Recall: {eval.recall(y_test, class_test_y)}')
print(f'Precision: {eval.precision(y_test, class_test_y)}')
print(f'F-measure: {eval.f1(y_test, class_test_y)}')