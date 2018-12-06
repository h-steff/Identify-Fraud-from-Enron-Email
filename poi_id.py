# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
import matplotlib
sys.path.append("../tools/")
import numpy as np
import math

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary_log', 'long_term_incentive_log', 'total_payments_log', 'shared_receipt_with_poi',
                'to_poi_ratio', 'from_poi_ratio', 'messages_total', 'to_message_ratio',
                 'from_message_ratio' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Store all names (keys of dictionary) in a list
name_list = []
for key in data_dict:
    #print key
    name_list.append(key)

#print name_list
#print len(name_list) # 146 names in this dataset

#print data_dict['METTS MARK']

# Store all available features in a list
all_features = []

for k in data_dict['METTS MARK']:
    all_features.append(k)

#print all_features

#print len(all_features) # 21 features in total in the beginning
#print len(data_dict['METTS MARK'])

# Get the number of POIs in the dataset and their names

poi_list = []

for i in data_dict:
    if data_dict[i]['poi'] == 1:
        poi_list.append(i)

print poi_list # names of the POIs
print "POIs: ", len(poi_list) # 18 POIs in the dataset
print "Total number: ", len(data_dict) # Total number of people in the dataset

### Task 2: Remove outliers

no_people = 0

for i in data_dict:
    #print i
    no_people += 1

#print no_people

# 146 keys in the data_dict -> 2 outliers: Total and the travel agency in the park -> remove

# remove outlier entries "TOTAL" and "THE TRAVEL AGENCY IN THE PARK"

data_dict_clean = data_dict

outlier_1 = data_dict_clean.pop("TOTAL")
outlier_2 = data_dict_clean.pop("THE TRAVEL AGENCY IN THE PARK")

people_clean = 0

for a in data_dict_clean:
    #print a
    people_clean += 1

#print people_clean

# identify entries where more than a certain percentage of features has value "NaN"

my_dataset_clean = data_dict_clean

total_features = []

for m in my_dataset_clean['METTS MARK']:
    total_features.append(m)

cutoff = 0.75

outlier_list = []


for person in my_dataset_clean:
    nan_features = []
    for feat in my_dataset_clean[person]:
        if my_dataset_clean[person][feat] == "NaN" or \
        my_dataset_clean[person][feat] == np.nan:
            nan_features.append(feat)
    #print nan_features
    nan_ratio = float(len(nan_features))/float(len(total_features))
    #print nan_ratio
    if nan_ratio >= cutoff:
        outlier_list.append(person)


#print total_features
#print len(total_features)
print "\nOutliers: ", outlier_list
print len(outlier_list)

poi_list_2 = []

for out in outlier_list:
    if my_dataset_clean[out]['poi'] == 1:
        poi_list_2.append(out)

print "POIs in outliers: ", poi_list_2

# at a cutoff of 0.75 there are 22 people where >= 75% of features have a
# value of "NaN", none of those 22 people is a known POI

# remove entries where more than a certain percentage of features has value "NaN"


def remove_outliers(data, outliers):
    data_out = data
    for x in outliers:
        if x in data_out:
            data_out.pop(x)
    return data_out

cleaned_data = remove_outliers(my_dataset_clean, outlier_list)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = cleaned_data
feat_list = ['salary', 'total_payments', 'long_term_incentive']

def feature_log10(dataset, features):
    dataset_log = dataset

    features_log = features

    for feature in features_log:

        for key in dataset_log:
        #print key
            if my_dataset[key][feature] != 'NaN':
                feature_log = math.log10(float(my_dataset[key][feature]))
                dataset_log[key][str(feature)+'_log'] = feature_log
            else:
                dataset_log[key][str(feature)+'_log'] = 'NaN'
    return dataset_log

dataset_log10 = feature_log10(my_dataset, feat_list)

#print dataset_log10['METTS MARK']
print "No. of features in dataset: ", len(dataset_log10['METTS MARK'])

# new feature: ratio of from_messages / to_messages from total messages

my_dataset_v1 = dataset_log10

for key in my_dataset_v1:
    #print key
    if my_dataset_v1[key]['to_messages'] != 'NaN' and \
    my_dataset_v1[key]['from_messages'] != 'NaN':
        messages_total = float(my_dataset_v1[key]['to_messages'] + \
        my_dataset_v1[key]['from_messages'])
        to_message_ratio = my_dataset_v1[key]['to_messages'] / messages_total
        from_message_ratio = my_dataset_v1[key]['from_messages'] / \
        messages_total

        my_dataset_v1[key]['messages_total'] = messages_total
        my_dataset_v1[key]['to_message_ratio'] = to_message_ratio
        my_dataset_v1[key]['from_message_ratio'] = from_message_ratio

    else:
        my_dataset_v1[key]['messages_total'] = 'NaN'
        my_dataset_v1[key]['to_message_ratio'] = 'NaN'
        my_dataset_v1[key]['from_message_ratio'] = 'NaN'

#print my_dataset_v1['METTS MARK']
print "No. of features in dataset: ", len(my_dataset_v1['METTS MARK'])

# new feature: ratio of messages from/to POI in to/from messages

my_dataset_v2 = my_dataset_v1

for key in my_dataset_v2:
    #print key
    if my_dataset_v2[key]['to_messages'] != 'NaN' and \
    my_dataset_v2[key]['from_messages'] != 'NaN':
        if my_dataset_v2[key]['from_poi_to_this_person'] != 'NaN' and \
        my_dataset_v2[key]['from_this_person_to_poi'] != 'NaN':
            if my_dataset_v2[key]['to_messages'] != 0 and \
            my_dataset_v2[key]['from_messages'] != 0:
                to_poi_ratio = my_dataset_v2[key]['from_poi_to_this_person'] / \
                float(my_dataset_v2[key]['to_messages'])
                from_poi_ratio = my_dataset_v2[key]['from_this_person_to_poi'] \
                 / float(my_dataset_v2[key]['from_messages'])


                my_dataset_v2[key]['to_poi_ratio'] = to_poi_ratio
                my_dataset_v2[key]['from_poi_ratio'] = from_poi_ratio
            else:
                my_dataset_v2[key]['to_poi_ratio'] = 'NaN'
                my_dataset_v2[key]['from_poi_ratio'] = 'NaN'

        else:
                my_dataset_v2[key]['to_poi_ratio'] = 'NaN'
                my_dataset_v2[key]['from_poi_ratio'] = 'NaN'
    else:
                my_dataset_v2[key]['to_poi_ratio'] = 'NaN'
                my_dataset_v2[key]['from_poi_ratio'] = 'NaN'

#print my_dataset_v2['METTS MARK']
print "No. of features in dataset: ", len(my_dataset_v2['METTS MARK'])

# new feature: long_term_incentive / total_payments -> incentive_total_ratio

my_dataset_v3 = my_dataset_v2


for key in my_dataset_v3:
    #print key
    #print "incentive: ", my_dataset_v3[key]['long_term_incentive']
    #print type(my_dataset_v3[key]['long_term_incentive'])
    #print "payments: ", my_dataset_v3[key]['total_payments']
    #print type(my_dataset_v3[key]['total_payments'])
    if my_dataset_v3[key]['long_term_incentive'] != 'NaN' and \
    my_dataset_v3[key]['total_payments'] != 'NaN':
        if my_dataset_v3[key]['long_term_incentive'] !=0 and \
        my_dataset_v3[key]['total_payments'] != 0:

            incentive_total_ratio = \
            float(my_dataset_v3[key]['long_term_incentive']) / \
            float(my_dataset_v3[key]['total_payments'])
            my_dataset_v3[key]['incentive_total_ratio'] = incentive_total_ratio

        else:
            my_dataset_v3[key]['incentive_total_ratio'] = 'NaN'

    else:
        my_dataset_v3[key]['incentive_total_ratio'] = 'NaN'


    #print my_dataset_v3[key]['incentive_total_ratio']
    #print type(my_dataset_v3[key]['incentive_total_ratio'])

#print my_dataset_v3['METTS MARK']
print "No. of features in dataset: ", len(my_dataset_v3['METTS MARK'])

# print all existing features

#for feature in features_list_all:
    #print "\n" + feature

# new feature: bonus / total_payments -> bonus_total_ratio

my_dataset_v4 = my_dataset_v3

for key in my_dataset_v4:
    #print key
    if my_dataset_v4[key]['bonus'] != 'NaN' and \
    my_dataset_v4[key]['total_payments'] != 'NaN':
        if my_dataset_v4[key]['bonus'] != 0 and \
        my_dataset_v4[key]['total_payments'] != 0:

            bonus_total_ratio = float(my_dataset_v4[key]['bonus']) / \
            float(my_dataset_v4[key]['total_payments'])
            my_dataset_v4[key]['bonus_total_ratio'] = bonus_total_ratio
            #print math.isinf(bonus_total_ratio)

        else:
            my_dataset_v4[key]['bonus_total_ratio'] = 'NaN'

    else:
        my_dataset_v4[key]['bonus_total_ratio'] = 'NaN'


#print my_dataset_v4['COLWELL WESLEY']
print "No. of features in dataset: ", len(my_dataset_v4['COLWELL WESLEY'])

# print all existing features

#for feature in features_list_all:
    #print "\n" + feature

# new feature: bonus / salary -> bonus_salary_ratio

my_dataset_v5 = my_dataset_v4

for key in my_dataset_v5:
    #print key
    if my_dataset_v5[key]['bonus'] != 'NaN' and \
    my_dataset_v5[key]['salary'] != 'NaN':
        if my_dataset_v5[key]['bonus'] != 0 and \
        my_dataset_v5[key]['salary'] != 0:

            bonus_salary_ratio = float(my_dataset_v5[key]['bonus']) / \
            float(my_dataset_v5[key]['salary'])
            my_dataset_v5[key]['bonus_salary_ratio'] = bonus_salary_ratio
            #print math.isinf(bonus_salary_ratio)

        else:
            my_dataset_v5[key]['bonus_salary_ratio'] = 'NaN'

    else:
        my_dataset_v5[key]['bonus_salary_ratio'] = 'NaN'


#print my_dataset_v5['COLWELL WESLEY']
print "No. of features in dataset: ", len(my_dataset_v5['COLWELL WESLEY'])

# For each feature, identify the number of people, where this feature has a
# value of "NaN"

my_dataset_clean_2 = my_dataset_v5

total_features = []

for m in my_dataset_clean['METTS MARK']:
    total_features.append(m)

thresh = 0.75

# create a dict where each feature is a key

feature_dict = {}

for person in my_dataset_clean_2:
    for feat in my_dataset_clean_2[person]:
        feature_dict[feat] = {'NaNs': 0}



for person in my_dataset_clean_2:
    for feat in my_dataset_clean_2[person]:

        if my_dataset_clean_2[person][feat] == "NaN" or \
        my_dataset_clean_2[person][feat] == np.nan:
            #print feature_dict[feat]['NaNs']
            #count += 1
            #print "Count2: ", count
            feature_dict[feat]['NaNs'] += 1

#print feature_dict

# feature_dict holds the name of the feature and the number of times this
# feature has a value of "NaN"

# add ratio of NaNs for each feature

dataset = my_dataset_v5

#print len(my_dataset_v5)

for feature in feature_dict:
    feature_dict[feature]['NaN_ratio'] = \
    float(feature_dict[feature]['NaNs'])/float(len(dataset))

#print feature_dict

for feature in feature_dict:
    print "\nFeature: ", feature
    print "NaN ratio: ", feature_dict[feature]['NaN_ratio']


### Split dataset

from sklearn.cross_validation import StratifiedShuffleSplit

my_dataset_t = my_dataset_v5

features_list_all = ['poi', 'to_messages', 'deferral_payments', 'expenses', 'deferred_income',
                     'long_term_incentive', 'from_message_ratio', 'salary_log', 'restricted_stock_deferred',
                     'messages_total', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other',
                     'to_poi_ratio', 'to_message_ratio', 'director_fees', 'bonus', 'total_stock_value',
                     'from_poi_to_this_person', 'from_this_person_to_poi', 'total_payments_log',
                     'from_poi_ratio', 'restricted_stock', 'long_term_incentive_log', 'salary', 'total_payments',
                     'exercised_stock_options', 'bonus_salary_ratio', 'incentive_total_ratio', 'bonus_total_ratio']

features_list_selector_9 = ['poi', 'from_message_ratio', 'messages_total', 'total_stock_value', 'bonus',
                             'total_payments_log',
                            'long_term_incentive_log', 'bonus_total_ratio', 'exercised_stock_options',
                            'incentive_total_ratio']

features_list_selector_3c = ['poi','deferred_income', 'loan_advances', 'bonus', 'total_stock_value', 'from_poi_ratio',
                             'salary', 'exercised_stock_options', 'bonus_salary_ratio', 'bonus_total_ratio']

features_list_selector_4 = ['poi','total_stock_value', 'bonus', 'total_payments_log', 'exercised_stock_options']

features_list_selector_4m = ['poi','expenses', 'shared_receipt_with_poi', 'other', 'bonus']

features_list_selector_6 = ['poi','bonus', 'total_stock_value', 'from_poi_ratio', 'exercised_stock_options',
                            'bonus_salary_ratio', 'bonus_total_ratio']

features_list_selector_6m = ['poi','deferral_payments', 'expenses', 'shared_receipt_with_poi', 'other', 'bonus',
                             'from_poi_ratio']


features_list_selector_9m = ['poi','deferral_payments', 'expenses', 'restricted_stock_deferred', 'shared_receipt_with_poi',
                             'other', 'to_poi_ratio', 'bonus', 'from_poi_ratio', 'long_term_incentive_log']

features_list_selector_12 = ['poi','deferred_income', 'shared_receipt_with_poi', 'loan_advances', 'bonus',
                             'total_stock_value', 'from_poi_ratio', 'salary', 'total_payments',
                             'exercised_stock_options', 'bonus_salary_ratio', 'incentive_total_ratio', 'bonus_total_ratio']

features_list_selector_12m = ['poi','deferral_payments', 'expenses', 'restricted_stock_deferred', 'shared_receipt_with_poi',
                              'other', 'to_poi_ratio', 'director_fees', 'bonus', 'total_stock_value', 'from_poi_ratio',
                              'restricted_stock', 'bonus_salary_ratio']


features_list_selector_15 = ['poi','expenses', 'deferred_income', 'long_term_incentive', 'salary_log',
                             'shared_receipt_with_poi', 'loan_advances', 'bonus', 'total_stock_value',
                             'from_poi_ratio', 'salary', 'total_payments', 'exercised_stock_options',
                             'bonus_salary_ratio', 'incentive_total_ratio', 'bonus_total_ratio']

features_list_selector_15m = ['poi','expenses', 'deferred_income', 'shared_receipt_with_poi', 'other', 'to_poi_ratio',
                              'bonus', 'total_stock_value', 'from_this_person_to_poi', 'from_poi_ratio',
                              'restricted_stock', 'total_payments', 'exercised_stock_options', 'bonus_salary_ratio',
                              'incentive_total_ratio', 'bonus_total_ratio']



data_dt = featureFormat(my_dataset_t, features_list_selector_9, sort_keys = True)
labels, features = targetFeatureSplit(data_dt)


# Split into training and test data

folds = 1000
cv = StratifiedShuffleSplit(labels, folds, random_state = 25,
        test_size = 0.15)

for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )



### select K-best features

# list of all (current) features

my_features = []

for k in my_dataset_v5['METTS MARK']:
    my_features.append(k)

#print my_features
#print len(my_features)

# important: 'poi' has to be the first feature!!
# remove email adress as this feature is a string and causes problems in the featureFormat fucntion
features_list_all = ['poi', 'to_messages', 'deferral_payments', 'expenses',
                    'deferred_income','long_term_incentive',
                    'from_message_ratio', 'salary_log',
                    'restricted_stock_deferred', 'messages_total',
                    'shared_receipt_with_poi', 'loan_advances', 'from_messages',
                    'other','to_poi_ratio', 'to_message_ratio', 'director_fees',
                    'bonus', 'total_stock_value','from_poi_to_this_person',
                    'from_this_person_to_poi', 'total_payments_log',
                    'from_poi_ratio', 'restricted_stock',
                    'long_term_incentive_log', 'salary', 'total_payments',
                    'exercised_stock_options', 'bonus_salary_ratio',
                    'incentive_total_ratio', 'bonus_total_ratio']


#print len(features_list_all)
# feature selection

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# create the selector

selector = SelectKBest(f_classif, k = 9)

selector.fit(features_train, labels_train)

#print "\n", selector.scores_
#print len(selector.scores_)

# get the selected k best features by first getting their indices
# calling .get_support get the features from list with all features by
# selecting their index
indexes = selector.get_support(indices=True)

#print indexes


k_features = [features_list_all[i+1] for i in indexes]

print "\nK-Best features: ", k_features


#print "features_train: ", features_train[0]

# reduce features_train to the k best features by transforming
features_train_transformed = selector.transform(features_train)

#print "\nfeatures_train_transformed: ", features_train_transformed[0]

# reduce features_test to the k best features by transforming
features_test_transformed = selector.transform(features_test)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Decision Tree

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

#########################################################
### your code goes here ###

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=11)
# try different parameters
clf_dt_1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=11,
            splitter='best')

t0 = time()
clf_dt_1.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" # taking time part 2

t1 = time() # timing the prediction
pred = clf_dt_1.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s" # prediction time part 2

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test)

print "Accuracy: ", acc

no_features = len(features_train[0])
print "No. of features: ", no_features


features_weight = clf_dt_1.feature_importances_

print "feature weight: ", features_weight

print "\ntester result: ", \
test_classifier(clf_dt_1, my_dataset_t, features_list_selector_9, folds=1000)

### Naive Bayes

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()


# rescale features

#scaled_train = scale_features(features_nb_train)
#scaled_test = scale_features(features_nb_test)

t0 = time() # use checking how long fitting the classifier takes
clf_nb.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" # taking time part 2

t1 = time() # timing the prediction
pred = clf_nb.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s" # prediction time part 2

#print pred
from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test) # accuracy goes down from 98,4% to 88,4% when slicing training set to 1%

print acc

print "\ntester result: ", test_classifier(clf_nb, my_dataset_t, features_list_selector_9, folds=1000)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# create a pipeline with PCA and classifier (Naive Bayes or Decision tree)

from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components=3)

# Classifier
clf_dt_p = tree.DecisionTreeClassifier(min_samples_split=10)

clf_NB_p = GaussianNB()

estimators = [('PCA', pca), ('clf', clf_dt_p)]
pipe = Pipeline(estimators)
pipe.fit(features_train, labels_train)

pred = pipe.predict(features_test)


print "\nPCA - explained variance: ", pca.explained_variance_ratio_

first_pc = pca.components_[0]

#print "\nFirst PC: ", first_pc

print "\ntester result: ", \
test_classifier(pipe, my_dataset_t, features_list_all, folds=1000)

# Best settings for NB: PCA - n_components = 8
# Best settings for DT: min_samples_split = 10; PCA - n_components = 3
#

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump decision tree classifier

clf_sub = clf_dt_1
my_dataset_sub = my_dataset_v5
features_list_sub = features_list_selector_9


dump_classifier_and_data(clf_sub, my_dataset_sub, features_list_sub)
