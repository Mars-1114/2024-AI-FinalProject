import numpy as np
from sklearn.ensemble import BaggingClassifier
import sklearn.discriminant_analysis
import sklearn.naive_bayes
import sklearn.svm
import sklearn.metrics

import eval_dict
import eval_stat
import eval_info
from utils import LOOKUP

# // PART A // BAGGING
#   Bootstrap Aggregating tries to lower the overfitting issue and increase the prediction accuracy.
#   It works by sampling the same dataset many times, training multiple "weak" classifier, then
#   compare the results (e.g. majority) of these to form a "strong" classifier.


