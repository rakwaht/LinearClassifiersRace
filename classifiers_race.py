#!/usr/bin/python
import sys
import time
start_time = time.time()

import numpy as np

n_samples=1000

# generate a random binary classification problem
from sklearn import datasets
dataset = datasets.make_classification(
	n_samples=n_samples,
	n_features=10,
	n_informative=2,
	n_repeated=0,
	n_classes=2
)
#print dataset

dataTrain = dataset[0]
dataTarget = dataset[1]

#k-fold cross validation
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
kf = cross_validation.KFold(n_samples, n_folds=10, shuffle=True, random_state=None)
# Index note: 0=>SVM, 1=>naive bayes, 2=>Forrest
accuracy = [[],[],[]]
f1 = [[],[],[]]
auc_roc = [[],[],[]]
for train_index, test_index in kf:
	X_train, X_test = dataTrain[train_index], dataTrain[test_index]
	Y_train, Y_test = dataTarget[train_index], dataTarget[test_index]
	
	# naive bayes classifier
	clf = GaussianNB()
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	accuracy[1].append(metrics.accuracy_score(Y_test, pred))
	f1[1].append(metrics.f1_score(Y_test, pred))
	auc_roc[1].append(metrics.roc_auc_score(Y_test, pred))

	# SVM classifier
	inner_n = len(X_train)
	C_Values = [1e-1, 1e-2, 1e-0, 1e1, 1e2]
	inner_score = []
	#Estimate best parameter
	for C in C_Values:
		clf = SVC(C=C, kernel='rbf', class_weight="balanced", random_state=None)
		inner_kf = cross_validation.KFold(inner_n, n_folds=5, shuffle=True, random_state=None)
		inner_f1 = []
		for inner_train_index, inner_test_index in inner_kf:
			X_inner_train, X_inner_test = X_train[inner_train_index], X_train[inner_test_index]
			Y_inner_train, Y_inner_test = Y_train[inner_train_index], Y_train[inner_test_index]
			clf.fit(X_inner_train, Y_inner_train);
			inner_pred = clf.predict(X_inner_test);
			# save F1 of internal cross folder validation
			inner_f1.append(metrics.f1_score(Y_inner_test, inner_pred))
		# compute average value
		inner_score.append(sum(inner_f1)/len(inner_f1))
	# Now we can choose the C that get the best F1
	best_C = C_Values[np.argmax(inner_score)]
	#print best_C
	# Prediction using best parameter predict labels for the test
	clf = SVC(C=best_C, kernel='rbf', class_weight="balanced", random_state=None)
	clf.fit(X_train, Y_train);
	pred = clf.predict(X_test);	
	# compute estimation parameters
	accuracy[0].append(metrics.accuracy_score(Y_test, pred))
	f1[0].append(metrics.f1_score(Y_test, pred))
	auc_roc[0].append(metrics.roc_auc_score(Y_test, pred))

	#Random Forest classifier
	inner_n = len(X_train)
	estimators_Values = [10, 100, 1000]
	inner_score = []
	#Estimate best parameter
	for estimator in estimators_Values:
		clf = RandomForestClassifier(n_estimators=estimator, criterion='gini', random_state=None)
		inner_kf = cross_validation.KFold(inner_n, n_folds=5, shuffle=True, random_state=None)
		inner_f1 = []
		for inner_train_index, inner_test_index in inner_kf:
			X_inner_train, X_inner_test = X_train[inner_train_index], X_train[inner_test_index]
			Y_inner_train, Y_inner_test = Y_train[inner_train_index], Y_train[inner_test_index]
			clf.fit(X_inner_train, Y_inner_train);
			inner_pred = clf.predict(X_inner_test);
			# save F1 of internal cross folder validation
			inner_f1.append(metrics.f1_score(Y_inner_test, inner_pred))
		# compute average value
		inner_score.append(sum(inner_f1)/len(inner_f1))
	# Now we can choose the C that get the best F1
	best_Estimator = estimators_Values[np.argmax(inner_score)]
	# Prediction using best parameter predict labels for the test
	clf = RandomForestClassifier(n_estimators=best_Estimator, criterion='gini', random_state=None)
	clf.fit(X_train, Y_train);
	pred = clf.predict(X_test);	
	# compute estimation parameters
	accuracy[2].append(metrics.accuracy_score(Y_test, pred))
	f1[2].append(metrics.f1_score(Y_test, pred))
	auc_roc[2].append(metrics.roc_auc_score(Y_test, pred))

print "Accuracy:"
sys.stdout.write ('Fold\tSVM\t\tNaive\t\tRndFrst\n')
print "-----------------------------------------------------------------"
for i in range(10): 
	sys.stdout.write ('%d\t%f\t%f\t%f\n' % (i, accuracy[0][i], accuracy[1][i], accuracy[2][i]))
print "-----------------------------------------------------------------"
sys.stdout.write ('\t%f\t%f\t%f\n' % (np.average(accuracy[0]), np.average(accuracy[1]), np.average(accuracy[2])))

print "F1:"
sys.stdout.write ('Fold\tSVM\t\tNaive\t\tRndFrst\n')
print "-----------------------------------------------------------------"
for i in range(10): 
	sys.stdout.write ('%d\t%f\t%f\t%f\n' % (i, f1[0][i], f1[1][i], f1[2][i]))
print "-----------------------------------------------------------------"
sys.stdout.write ('\t%f\t%f\t%f\n' % (np.average(f1[0]), np.average(f1[1]), np.average(f1[2])))

print "AucRoc:"
sys.stdout.write ('Fold\tSVM\t\tNaive\t\tRndFrst\n')
print "-----------------------------------------------------------------"
for i in range(10): 
	sys.stdout.write ('%d\t%f\t%f\t%f\n' % (i, auc_roc[0][i], auc_roc[1][i], auc_roc[2][i]))
print "-----------------------------------------------------------------"
sys.stdout.write ('%f\t%f\t%f\n' % (np.average(auc_roc[0]), np.average(auc_roc[1]), np.average(auc_roc[2])))

print("--- %s seconds ---" % (time.time() - start_time))
