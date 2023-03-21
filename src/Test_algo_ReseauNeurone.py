from sklearn.neural_network import MLPClassifier
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
    'hidden_layer_sizes': [ (100,), (100, 50),(200, 100)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': [500, 1000],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'momentum': [0.9, 0.95],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32],
    'early_stopping': [True],
    'validation_fraction': [0.2, 0.3],
    'beta_1': [0.9, 0.95],
    'beta_2': [0.99, 0.999],
    'shuffle': [True, False],
    'tol': [1e-4, 1e-5],
    'n_iter_no_change': [5, 10]
}

# Création de l'objet MLPClassifier
mlp = MLPClassifier(random_state=42)

# Recherche des meilleurs paramètres avec GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5,n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Affichage des meilleurs paramètres
print(grid_search.best_params_)
