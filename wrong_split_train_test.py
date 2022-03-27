from sklearn import naive_bayes
from sklearn import datasets

# un algo d'apprentissage: We train on the first 100 instances and test on the last 50
# => We only trained our model on the classes 0 and 1. All the predicted classes are 0/1
# Our model doesn't know the existance of the second class.

nb = naive_bayes.MultinomialNB(fit_prior=True)
irisData = datasets.load_iris()
#we train on the first 100 instances
nb.fit(irisData.data[:99], irisData.target[:99])
#we test on the last 50 instances
test = nb.predict(irisData.data[100:149])
print(test)
instance_31 = nb.predict([irisData.data[31]])
print(instance_31)
last_instance = nb.predict([irisData.data[-1]])
print(last_instance)
all_instances = nb.predict(irisData.data[:])
print(all_instances)
#print(irisData.target)