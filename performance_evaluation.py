from sklearn import naive_bayes
from sklearn import datasets
import numpy as np


nb = naive_bayes.MultinomialNB(fit_prior=True)

irisData = datasets.load_iris()
nb.fit(irisData.data[:], irisData.target[:])
Y = irisData.target
P = nb.predict(irisData.data[:])

#erreur d'apprentissage (method1)
ea = 0
for i in range(len(irisData.data)):
    if (P[i] != Y[i]):
        ea = ea+1
ea = ea /len(irisData.data)
print(f"error1 = {ea}")

#erreur d'apprentissage (method2)
ea2 = len(Y[(Y - P)!=0])/len(irisData.data)
print(f"error2 = {ea2}")


#erreur d'apprentissage (method3)
ea3 = np.count_nonzero(P-Y)/len(irisData.data)
print(f"error3 = {ea3}")


accuracy = nb.score(irisData.data, irisData.target)
print(f"accuracy = {accuracy}")
print(f"accuracy 2 (1-e) = {1-ea3}")


assert ea == ea2
assert ea == ea3
assert accuracy == 1-ea