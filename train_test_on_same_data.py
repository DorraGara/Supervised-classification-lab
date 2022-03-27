from sklearn import naive_bayes
from sklearn import datasets


# un algo d'apprentissage: We train all the instances and we test on the same instances.

#fit_prior :Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
nb = naive_bayes.MultinomialNB(fit_prior=True)
irisData = datasets.load_iris()
#train
nb.fit(irisData.data[:], irisData.target[:])
instance_31 = nb.predict([irisData.data[31]])
print(instance_31)
last_instance = nb.predict([irisData.data[-1]])
print(last_instance)
all_instances = nb.predict(irisData.data[:])
print(all_instances)
#print(irisData.target)