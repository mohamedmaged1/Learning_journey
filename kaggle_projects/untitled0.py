import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


train_data = pd.read_csv('train.csv')
test_data =  pd.read_csv('test.csv')

train_lbl=train_data['target']
train_data=train_data.drop(['ID_code','target'],axis=1)

from sklearn import decomposition

## PCA decomposition
pca = decomposition.PCA(n_components=180) #Finds first 180 PCs
pca.fit(train_data)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of variance explained')

pca = decomposition.PCA(n_components=100) 
pca.fit(train_data)
PCtrain = pd.DataFrame(pca.transform(train_data))

max(PCtrain.max())
PCtrain=PCtrain/max(PCtrain.max())


#Import Libraries
from sklearn.model_selection import train_test_split
#----------------------------------------------------

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(PCtrain, train_lbl, test_size=0.33, random_state=44, shuffle =True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

n_neig=np.arange(2,10)
acc=[]
for i in n_neig:
    KNNModel = KNeighborsClassifier(n_neighbors= i)
    KNNModel.fit(X_train[::10], y_train[::10])
    acc.append(KNNModel.score(X_train, y_train))
    
plt.plot (n_neig,acc)
plt.xlabel('neighbours')
plt.xlim(2,20)
plt.ylabel('accuracy')
plt.show()

#as we 've seen at neighbours=4 best 
import sklearn.externals.joblib as jb


model = KNeighborsClassifier(n_neighbors= 4)
model.fit(X_train[::10], y_train[::10])

jb.dump(model , 'saved file.sav')


##############################################

savedmodel = jb.load('saved file.sav')
#############################################
print('KNNClassifierModel Train Score is : ' , model.score(X_train, y_train))
print('KNNClassifierModel Test Score is : ' , model.score(X_test, y_test))
