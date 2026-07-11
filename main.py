from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split 

X,Y = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=2,
    random_state=42
)

X_train ,X_test ,Y_train , Y_test = train_test_split(X,Y ,random_state=42)

clf = Perceptron(
    random_state=42 , 
    max_iter=1000,
    eta0=1.0,
    tol=1e-3 , 
    shuffle=True
)

clf.fit(X_train,Y_train)

Accuracy = clf.score(X_test,Y_test)

print(Accuracy)
