from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
    'n_estimators': [100, 200, 300],
    'criterion': ['entropy','gini'],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': [5, 10, 15]
}

# Création de l'objet RandomForestClassifier
rfc = RandomForestClassifier()

# Recherche des meilleurs paramètres avec GridSearchCV
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Affichage des meilleurs paramètres
print(grid_search.best_params_)