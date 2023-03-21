from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd

#Import du dataset à tester
data = pd.read_csv("data/training_dataset")
data.drop(457, axis=0, inplace=True)
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]

#Répartition 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Définition des paramètres à tester

param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'degree': [2, 3],
    'coef0': [0, 1],
    'shrinking': [True, False],
    'tol': [1e-3, 1e-4],
    'class_weight': [None, 'balanced']
}



# Création de l'objet SVC
svm = SVC(random_state=42)

# Recherche des meilleurs paramètres avec GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Affichage des meilleurs paramètres
print(grid_search.best_params_)
