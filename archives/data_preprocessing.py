# Data Preprocessing

# Importer les librairies
import pandas as pd
import numpy as np

# Importer le dataset (ne pas oublier de parametrer le "console working 
# directory")
dataset = pd.read_csv('Data.csv')
#print(dataset.head())
# iloc prend deux parametres : nombre de lignes et nombre de colonnes
# ":" signifie que l'on prend toutes les lignes ou colonnes du dataset
# en faisant ":-1" on part du principe que la derniere colonne contient la 
# colonne de variables dependantes que l'on mettra dans le vecteur y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # on recupere juste la derniere colonne

# Gérer les données manquantes en important la class Imputer du module 
# preprocessing de la librairie scikit-learn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# on lie l'imputer a notre variable en precisant les lignes puis colonnes 
# que l'on souhaite. Pour rappel, dans un range, la borne superieure est 
# toujours exclue en python donc 1:3 correspond aux colonnes dont l'indice 
# est 1 et 2
imputer.fit(X[:, 1:3])
# on transforme la variable conformement a la strategie definie 
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("Independent dataset X:")
print(X)
print("")

# Gérer les variables catégoriques (ie variable non numerique continue)
# ici dans notre dataset, les pays (3 categories) et purchase (2 categories)
# Le but est d'encoder les variables catégoriques (en texte par exemple) en
# variable numerique
# Variable nominale (= non ordonnee) vs. ordinale (= ordonnee)
# La class LabelEncoder permet de transformer du texte en valeur numérique 
# (1, 2, 3, ...)
# La class OneHotEncoder permet de spliter en plusieurs colonnes une colonne
# contenant des valeurs numeriques (= pas de texte)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # pour la variable country
# fit_transform fait d'abord fit puis transform
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # country est la 1ere colonne
# Méthode d'encodage pour une variable catégorique nominale = encode par 
# dummy-variable (obligatoire pour les variables independantes = entrees)
# L'encodage par dummy-variable consiste a decouper en N colonnes une variable
# categorique à N categories (ie. valeurs distinctes) puis de mettre 1 ou 0
# dans chacune de ces colonnes (par exemple la valeur France de la colonne
# country est remplacée par un 1 dans la colonne France et 0 dans les colonnes
# Germany et Spain). Si on avait simplement remplacé les pays par un chiffre 
# (0 pour la France, 1 pour Germany et 2 pour Spain), on aurait recréé une 
# relation d'ordre entre les pays puisque 1 < 2 etc.
# categorical_features correspond aux indices des colonnes a transformer
onehotencoder = OneHotEncoder(categorical_features = [0], dtype=int) 
np.set_printoptions(suppress=True)
X = onehotencoder.fit_transform(X).toarray()
print("Independent dataset X apres gestion des variables categoriques:")
print(X)
print("")
# pour la variable purchase on transforme juste les valeurs textuelles en 
# valeurs numeriques
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y)

# Diviser le dataset entre le Training set et le Test set
# le training set est la base d'apprentissage pour découvrir les corrélations
# entre les variables independantes (entrees) et les variables dependantes 
# (sorties) afin d'eviter le sur-apprentissage (apprentissage par coeur), on 
# conserve une partie des donnees pour tester le modele d'apprentissage avec 
# les valeurs attendues
# Les observations du test set n'auront pas contribué à l'apprentissage et on
# dimensionne a 20% des observations totales, le nombre d'observations à mettre
# de coté pour le test set
# si la precision (taux d'erreur) du test set est significativement plus
# elevee que le train set, cela signifie qu'il y a eu sur-apprentissage
# On importe directement la fonction train_test_split
from sklearn.model_selection import train_test_split
# on renseigne le random_state pour est sur d'avoir les mêmes resultats a 
# chaque execution : utile uniquement dans un but pedagogique ou de debugging
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# on met a la meme echelle toutes les variables pour eviter qu'une variable
# d'entrée n'écrase toutes les autres (par exemple, dans notre dataset la
# colonne Age n'a pas du tout la même échelle que la colonne Salaire)
from sklearn.preprocessing import StandardScaler
# on utilise ici la méthode de standardisation (qui consiste a soustraire
# aux valeurs de x la moyenne des valeurs de x le tout divisé par l'écart-type
# de x)
# une autre methode est la methode de normalisation (qui consiste a soustraire
# aux valeurs de x la valeur minimum de x le tout divisé par la soustraction
# de la valeur maximum de x par la valeur minimum de x)
sc = StandardScaler()
#print(X_train)
X_train = sc.fit_transform(X_train)
# Grace la fonction train_test_split, le training set et le test set ont une
# moyenne et un ecart-type similaire et une distribution similaire des valeurs
# On n'a donc pas besoin de fitter sc à X_test puisqu'il est déjà fitté à 
# X_train
X_test = sc.transform(X_test)
#print(X_train)
# On n'applique par le feature scaling sur y (sorties) puisque dans ce 
# dataset, les valeurs ont déjà été encodées (avec LabelEncoder) en 0 ou 1
# ce qui est dans une echelle similaire aux variables de X (entrees).
# En fonction du dataset, il peut être nécessaire d'appliquer le feature
# scaling aux sorties (typiquement si y est un vecteur de valeurs numériques
# avec une échelle très différentes des valeurs de X)