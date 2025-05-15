#Un réseau de neurone pour un pb de classification multiclasses.
#dataset=Boston housing prices (housing.csv)

import pandas as pn
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#Une fonction qui prend en entrée une table des prix
#et transforme chaque prix en l'une des valeurs Low,Intermediate, High.
def num2categ(tab, b1, b2):
   tab_ret = [] 
   for i in range(tab.size):
     if (tab[i]<=b1):
       tab_ret.append("Low")
     elif (tab[i]>=b2):
       tab_ret.append("High")
     else:
       tab_ret.append("Intermediate")
   return tab_ret
   
#Encoding classes

def encodeClass(s_class):
  if (s_class=='Low'):
    return [1,0,0]
  elif (s_class=='High'):
    return [0,0,1]
  else:
    return [0,1,0]

#reading the data
myData = pn.read_csv("housing.csv")
nbColumns = myData.shape[1]
nbVars = nbColumns
#Adding the column CLS
price = np.array(myData['Price'])
b1 = np.quantile(price, 0.15)
b2 = np.quantile(price, 0.85)

#Ajoute de la colonne CLS (class)

tab_cls = num2categ(price, b1, b2)
lst_cls = list(tab_cls)
myData['CLS'] = lst_cls
print("We have ", lst_cls.count('High')," High")
print("We have ", lst_cls.count('Low')," Low")
print("We have ", lst_cls.count('Intermediate')," Intermediate")

#Définition des ensembles d'apprentissage et de test
X=myData.values[:,:nbVars]
X=X.astype('float64')
Y=myData.values[:,nbColumns]
nbClasses = 3

encoded_Y = np.array([encodeClass(y) for y in Y])

X_train, X_test, Y_train, Y_test = train_test_split( X, encoded_Y, test_size = 0.3, random_state = 100)

#Création du réseau de neurones. La couche de sortie a un nombre de neurones
#égal au nombre de classes et softmax comme fonction d'activation. 

nn = Sequential()
nn.add(Dense(5, input_dim=nbVars, activation='sigmoid'))
nn.add(Dense(nbClasses, activation='softmax'))
nn.summary()
#Compléter les caractéristiques du RNA, notamment la loss function (ici 
#la categorical_crossentropy car c'est un pb de multiclassification) et la
#métriques utilisée pour mesurer sa performance (ici l'accuracy).
#https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#nn.fit(X_train, Y_train, epochs=500, batch_size=10)

score = nn.evaluate(X_test, Y_test, verbose=2)

print('Test accuracy:', score[1])

#Pour plus d'information, on construit la matrice de confusion.
Y_pred = nn.predict(X_test)
Y_pred_1 = Y_pred.argmax(axis=1)
Y_test_1 = Y_test.argmax(axis=1)
confusion = confusion_matrix(Y_pred_1, Y_test_1)
print(confusion)
