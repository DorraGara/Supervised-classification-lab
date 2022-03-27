import numpy
from sklearn import naive_bayes
from sklearn import datasets

def split(data,proportion):
    dataS1 = numpy.empty(shape=(0,numpy.shape(data.data)[1]))
    targetS1 = numpy.empty(shape=(0,))
    dataS2 = numpy.empty(shape=(0,numpy.shape(data.data)[1]))
    targetS2 = numpy.empty(shape=(0,))
    for i in set(data.target):
        class_data = data.data[data.target == i]
        m = round(proportion * len(class_data))
        numpy.random.permutation(class_data)
        dataS1= numpy.concatenate((dataS1,class_data[:m]),axis=0)
        class_target = numpy.repeat(i,m)
        targetS1= numpy.concatenate((targetS1,class_target),axis=0)
        dataS2= numpy.concatenate((dataS2,class_data[m:]),axis=0)
        class_target2 = numpy.repeat(i,len(class_data)-m)
        targetS2= numpy.concatenate((targetS2,class_target2),axis=0)
    return (dataS1,targetS1,dataS2,targetS2)


def test(data, clf,proportion):
    dataS1,targetS1,dataS2,targetS2 = split(data,proportion)
    clf.fit(dataS1, targetS1)
    accuracy = clf.score(dataS2, targetS2)
    error = 1 - accuracy
    return error

def test_iter(data,clf,iter,proportion):
    sum = 0
    for i in range(iter):
        sum += test(data,clf,proportion)
    return sum / iter

irisData = datasets.load_iris()
clf = naive_bayes.MultinomialNB(fit_prior=True)
proportion = 9/10

#the average error is stable => converge to a value.
iterations = [10,50,100,200,500,1000]
# for i in iterations:
#     error_moy = test_iter(irisData,clf,i,proportion)
#     print(f"average error for {i} iterations: {error_moy}")

from sklearn.model_selection import train_test_split
def test2(data, clf,proportion):
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size=proportion)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    error = 1 - accuracy
    return error

error = test(irisData,clf,proportion)
print(f"error = {error}")

error = test2(irisData,clf,proportion)
print(f"error library split = {error}")