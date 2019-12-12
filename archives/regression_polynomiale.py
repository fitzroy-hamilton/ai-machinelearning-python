# Regression Polynomiale

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Construction du modèle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# degre a adapter pour matcher au mieux aux observations
# plus le degre est eleve, plus le risque de sur-apprentissage est eleve
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
polynomialRegressor = LinearRegression()
polynomialRegressor.fit(X_poly, y)

# juste pour comparaison avec une regression lineaire sur le graphique
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)

# Faire de nouvelles prédictions

# Visualiser les résultats
plt.scatter(X, y, color = 'red')
plt.plot(X, polynomialRegressor.predict(X_poly), color = 'blue')
plt.plot(X, linearRegressor.predict(X), color = 'orange') # pour comparaison
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()

# Visualiser les résultats (courbe plus lisse)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, polynomialRegressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.plot(X, linearRegressor.predict(X), color = 'orange') # pour comparaison
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()