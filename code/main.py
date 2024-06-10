import numpy as np
import pandas as pd
import re
import dask
import dask.diagnostics
import dask.diagnostics.progress
import random
from statistics import multimode
import pickle
from pathlib import Path
import matplotlib
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import eval_dict
import eval_stat
import eval_info
from utils import LOOKUP

cwd = Path.cwd()
path = Path(__file__).parent
clfpath = (path / "../classifiers/").resolve()

## The comments describe the training process

# PART A // DATASET SPLITTING
#   In data analysis, it it best to have a evaluation for our trained model. This is most commonly
#   by splitting the original dataset into training set and test set (sometimes validation set).
def split_data(file, ratio = 0.8):
    '''
    ## Perform a train-test split
        // arguments //
            file - <str> - name of the csv file\n
            ratio - <float> - test-train split ratio, default 0.8
        
        // return //
            this function does not return anything
    '''
    data = pd.read_csv(file)
    train = data.sample(frac = ratio)
    test = data.drop(train.index)
    
    train.set_index('labels').to_csv("training.csv")
    test.set_index('labels').to_csv("test.csv")

# PART B // LOOKUP TABLE
#   In this project, we will use dictionary and frequency table to compute the feature evaluation.
#   This needs to be done before bagging because the later testing process also needs to use these tables.

def build_table(dataset):
    '''
    ## Build the lookup table
        // argument //
            dataset - <str> - name of the csv file
        // return //
            this function does not return anything
    '''
    global dictionary
    global letter_freq
    global twoGram_freq
    
    dictionary = eval_dict.dict_table(dataset)
    letter_freq = eval_stat.char_freq_table(dataset)
    twoGram_freq = eval_stat.twog_freq_table(dataset)

# PART C // BUILD FEATURE
#   In classification model, the input is usually a list of features and a tag representing its category (in our
#   case, language). This is why we designed various evaluation functions.

def build_features(txt):
    '''
    ## Transform the data into a feature list
    the list contains a whopping 62 features 
    (20 from dictionary search, 40 from frequency analysis, and 2 from information density)
        // arguments //
            txt - <string> - the input text\n
            dict_table - <dict> - a table contains every letter appears in different languages
                format: {language id : list of chars}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
            char_freq_table - <dict> - a table that list out the frequency rank of letters for all languages
                format: {language id : list of chars}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
            twoGram_freq_table - <dict> - a table that list out the frequency rank of 2-grams for all languages
                format: {language id : list of 2-grams}
                e.g. {'en': ['ab', 'gh', 'ck', ...], ...}
        
        // return //
            feature - <list> - list of floats that contains 62 features
    '''
    feature = []
    evalDict = eval_dict.lang_percent(txt, dictionary)
    evalCharFreq = eval_stat.char_freq_corr(txt, letter_freq)
    evalTwoGramFreq = eval_stat.twog_freq_corr(txt, twoGram_freq)
    evalWordLen = eval_info.avg_word_len(txt)
    evalSentenceLen = eval_info.avg_sent_len(txt)
    
    feature.extend([evalDict[key] for key in LOOKUP])
    feature.extend([evalCharFreq[key] for key in LOOKUP])
    feature.extend([evalTwoGramFreq[key] for key in LOOKUP])
    feature.append(evalWordLen)
    feature.append(evalSentenceLen)
    
    return feature

# PART D // PREMAKE FEATURE TABLE
#   Due to the sheer size of the lookup table, the feature constructing process turns out taking
#   a lot of time (est. 6 hours). Although this can be optimised by parallel computation, it's still
#   extremely time-consuming. To solve this, we can prebuild the feature table for later use.
def feature_table():
    '''
    ## Build the feature table for both training and test set
    '''
    train_dataset = pd.read_csv("training.csv").reset_index()
    test_dataset = pd.read_csv("test.csv").reset_index()
    x_train = []
    y_train = []
    x_test = []
    y_test =[]
    lazyx_train = []
    lazyx_test = []
    
    # build training feature
    print("Computing training feature")
    for index, data in train_dataset.iterrows():
        resultx = dask.delayed(build_features)(data["text"])
        lazyx_train.append(resultx)
        y_train.append(data["labels"])
    # parallel processing to speed up
    with dask.diagnostics.progress.ProgressBar():
        x_train = dask.compute(*lazyx_train)
    
    # build test feature
    print("Computing testing feature")
    for index, data in test_dataset.iterrows():
        resultx = dask.delayed(build_features)(data["text"])
        lazyx_test.append(resultx)
        y_test.append(data["labels"])
    # parallel processing to speed up
    with dask.diagnostics.progress.ProgressBar():
        x_test = dask.compute(*lazyx_test)
    
    feature_train = [x_train, y_train]
    feature_test = [x_test, y_test]
    
    # export lists
    with open("feat_train", "wb") as file:
        pickle.dump(feature_train, file)
    
    with open("feat_test", "wb") as file:
        pickle.dump(feature_test, file)
    
# PART E // BAGGING
#   Bootstrap Aggregating tries to lower the overfitting issue and increase the prediction accuracy.
#   It works by sampling the same dataset many times, training multiple "weak" classifier, then
#   compare the results (e.g. majority) of these to form a "strong" classifier.
def bagging(algorithm, size = 10, ratio = 0.63, features = list(range(62))):
    '''
    ## Training the model by using bagging technique and various algorithms
        // arguments //
            algorithm - <sklearn.classifier> - the classification model\n
            size - <int> - the number of weak classifier, default 10\n
            ratio - <float> - the sample proportion, default 0.63
            features - <list> - 
        // return
            this function does not return anything
    '''
    # import feature table
    with open("feat_train", "rb") as file:
        train = pickle.load(file)
    with open("feat_test", "rb") as file:
        test = pickle.load(file)
    
    # train model
    train_feature = list(train[0])
    train_tag = train[1]
    
    # take specific features for training
    for n in range(len(train_feature)):
        train_feature[n] = [train_feature[n][x] for x in features]
    
    classifiers = []    # store classifiers for later bagging
    for n in range(size):
        print("Training Bag {}/{}".format(n + 1, size))
        # sample the training set (again)
        index = random.sample(range(len(train_feature)), round(len(train_feature) * ratio))
        feature_sampled = [train_feature[x] for x in index]
        tag_sampled = [train_tag[x] for x in index]
        
        if algorithm == LinearSVC:
            clf = LinearSVC(multi_class = "ovr")
        else:
            clf = algorithm()
        clf.fit(feature_sampled, tag_sampled)
        classifiers.append(clf)
    
    # export classifiers
    with open((clfpath / algorithm.__name__).resolve(), "wb") as file:
        pickle.dump(classifiers, file)

# PART F // ACCURACY
#   After training the model, we should test its accuracy. Since we're using bagging, we take the majority.
def accuracy(algorithm, features = list(range(62))):
    '''
    ## Test the accuracy of the given model. This also returns the prediction results of bagging.
    It computes both training set and test set, and single and combined classifiers.
        // argument //
            algorithm - <sklearn.classifier> - the classification model
            
        // returns //
            new_predict_train - <list> - a list of predicted training set tags
            new_predict_tag - <list> - a list of predicted test set tags
    '''
    # import model
    with open((clfpath / algorithm.__name__).resolve(), "rb") as file:
        model = pickle.load(file)

    # import feature table
    with open("feat_train", "rb") as file:
        train = pickle.load(file)
    with open("feat_test", "rb") as file:
        test = pickle.load(file)
        
    train_feature = list(train[0])
    train_tag = train[1]
    test_feature = list(test[0])
    test_tag = test[1]
    
    # take specific features
    for n in range(len(train_feature)):
        train_feature[n] = [train_feature[n][x] for x in features]
    for n in range(len(test_feature)):
        test_feature[n] = [test_feature[n][x] for x in features]
    
    # check the accuracy of single classifiers
    predict_train = []
    predict_test = []
    for n in range(len(model)):
        predict_train.append(model[n].predict(train_feature))
        predict_test.append(model[n].predict(test_feature))
        
        # print accuracy
        acc_train = round(accuracy_score(train_tag, predict_train[n]), 6)
        acc_test = round(accuracy_score(test_tag, predict_test[n]), 6)
        print("Classifier {}/{}: Train Accuracy = {}, Test Accuracy = {}".format(n + 1, len(model), acc_train, acc_test))
    print("")
    
    # check the accuracy of different number of classifiers
    for n in range(len(model)):
        i = n + 1
        # check every two classifiers
        if (i % 2 == 0 and i > 0) or i == len(model):
            new_predict_train = []
            new_predict_test = []
            for x in range(len(predict_train[0])):
                majority = multimode([clf[x] for clf in predict_train])
                new_predict_train.append(random.sample(majority, 1)[0])
            for x in range(len(predict_test[0])):
                majority = multimode([clf[x] for clf in predict_test])
                new_predict_test.append(random.sample(majority, 1)[0])
                
            acc_train = round(accuracy_score(train_tag, new_predict_train), 6)
            acc_test = round(accuracy_score(test_tag, new_predict_test), 6)
            print("Take {} classifiers: Train Accuracy = {}, Test Accuracy = {}".format(i, acc_train, acc_test))
    
    return new_predict_train, new_predict_test

# PART G // VOTING
#   This way it work is by a majority vote as well. But unlike bagging, voting takes the results
#   from -different- classifiers, which may give us a different outcome
def voting(predict_table, true_table):
    '''
    ## Perform a voting classifier and check the accuracy
        // argument //
            predict_table - [list, list, ...] - list of predictions from different classifiers
            true_table - <list> - list of actual tags of the sample
            
        // return //
            acc - <float> - the accuracy of this classifier
    '''
    prediction = []
    for n in range(len(predict_table[0])):
        majority = multimode([x[n] for x in predict_table])
        prediction.append(random.sample(majority, 1)[0])
    
    acc = round(accuracy_score(true_table, prediction), 6)
    return acc

# PART H // USER TESTING
#   This is the ultimate goal of our project: Detect the language of any user input
def user_test(txt, clfs, features = list(range(62))):
    '''
    '''
    # text preprocess
    txt = re.sub(r'(?<!\.$)[^\w\s.!?。？！]', '', txt)
    txt = re.sub(r'(?<!\.$)[\d]', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    
    # build feature table
    feature = build_features(txt)
    feature = [feature[x] for x in features]
    
    # perform prediction
    model_predict = []
    for clf in clfs:
        # import model
        with open((clfpath / clf.__name__).resolve(), "rb") as file:
            model = pickle.load(file)
        
        predict_list = []
        for n in model:
            predict_list.append(n.predict([feature])[0])
        
        model_predict.append(random.sample(multimode(predict_list), 1)[0])
    
    vote_predict = random.sample(multimode(model_predict), 1)[0]
    print("Language: {}".format(LOOKUP[vote_predict]))
        
    
    

###################################################################
###################################################################

def main(new_split = False, remodel = True, scoring = True):
    '''
    ## Run the program.
        // argument //
            new_split - <bool> - Make a new train-test set. (est. 7 hours)\n
            remodel - <bool> - Rebuild the bagging model. (est. 8 mins)\n
            scoring - <bool> - Compute the accuracy score. (est. 30 mins)
            
        // return //
            this function does not return anything
    '''
    clfs = [QuadraticDiscriminantAnalysis, DecisionTreeClassifier, LinearSVC]
    features = list(range(20, 62))
    size = 10
    
    if new_split:
        print("Splitting test-train set...")
        split_data("cleaned_train.csv")
        
    print("Building lookup tables...")
    build_table("training.csv")
    
    if new_split:
        print("Building feature tables...")
        print("### WARNING ###")
        print("This will take a VERY LONG time to build (approx. 6~8 hrs)")
        print("###############")
        feature_table()
    
    
    train_predict_table = []
    test_predict_table = []
    
    if remodel:
        print("\n## Model will be trained with {} features and {} algorithms (Bag size: {}) ##".format(len(features), len(clfs), size))
        print("Training the classifier...")
        for clf in clfs:
            print("Algorithm: {}".format(clf.__name__))
            bagging(clf, size, features=features)
            print("\n")
    
    if scoring:
        print("Testing the model...")
        for clf in clfs:
            print("\n-[Algorithm: {}]-".format(clf.__name__))
            train, test = accuracy(clf, features=features)
            train_predict_table.append(train)
            test_predict_table.append(test)
    
        # prepare for voting
        print("\n-[Performing Voting Technique with {} Bagged Classifiers]-".format(len(clfs)))
        with open("feat_train", "rb") as file:
            train_tag = pickle.load(file)[1]
        with open("feat_test", "rb") as file:
            test_tag = pickle.load(file)[1]
        
        acc_train = voting(train_predict_table, train_tag)
        acc_test = voting(test_predict_table, test_tag)
        print("Train Accuracy: {}, Test Accuracy: {}".format(acc_train, acc_test))
    
    user_test("Hello everybody my name is Markiplier and today we're gonna play an indie horror game called Five Nights at Freddy's", clfs, features)
    
    
main()