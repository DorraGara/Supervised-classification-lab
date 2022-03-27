import numpy
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

irisData = datasets.load_iris()
clf = [naive_bayes.MultinomialNB(fit_prior=True), DecisionTreeClassifier()]
folds = [2,3,5,8,10]
for c in clf:
    print(f"clf:{c}")
    for i in folds:
        accuracy = cross_val_score(c, irisData.data, irisData.target, cv=i)
        average_accuracy = numpy.average(accuracy)
        error = 1 - average_accuracy
        print(f"error({i}):{error}")