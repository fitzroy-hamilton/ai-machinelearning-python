# Regression Lineaire Multiple

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Gerer les variables categoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# on retire une colonne correspondant à la dummy variable que
# l'on retire pour assurer l'independance des variables
X = X[:, 1:] 

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# pas besoin de feature scaling pour la regression multiple puisque les 
# coefficients peuvent s'adapter pour etre a la bonne echelle

# Construction du modele
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Faire de nouvelles predictions
y_pred = regressor.predict(X_test)
# contrairement a une regression lineaire simple, on passe en parametre
# a predict un tableau d'entrees correspondant aux multiples entrees de la
# regression lineaire multiple
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))