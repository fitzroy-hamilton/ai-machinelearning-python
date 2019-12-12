# Regression Lineaire Simple
# Hypothèses de la régression linéaire :
    # Exogénéité
    # Homoscédasticité
    # Erreurs indépendantes
    # Normalité des erreurs
    # Non colinéarité des variables

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Diviser le dataset entre le Training set et le Test set
# Il s'agit de la base d'apprentissage splittée entre apprentissage et test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1.0/3, random_state = 0)

# On n'applique par le feature scaling sur un modele de regression lineaire
# simple

# Construction du modele
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Application de la methode des moindres carrés ordinaires (minimisation de 
# l'erreur) pour trouver les bons paramètres du modèles de telle sorte que
# y = b_0 + b_1 * X
# Il s'agit en fait de l'apprentissage supervisé
regressor.fit(X_train, y_train)

# Faire de nouvelles predictions
y_pred = regressor.predict(X_test)
regressor.predict(15)

# Visualiser les resultats
plt.scatter(X_train, y_train, color = 'orange') # points de la base d'apprentissage
plt.scatter(X_test, y_test, color = 'red') # points de la base de test
plt.plot(X_train, regressor.predict(X_train), color = 'gray') # droite
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()