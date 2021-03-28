import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import xgboost as xgb

train_data = pd.read_csv('train.csv')
test_data =  pd.read_csv('test.csv')

train_lbl=train_data['target']
train_data=train_data.drop(['ID_code','target'],axis=1)
test_data=test_data.drop(['ID_code'],axis=1)

#from sklearn import decomposition
#
### PCA decomposition
#pca = decomposition.PCA(n_components=180) #Finds first 180 PCs
#pca.fit(train_data)
#plt.plot(pca.explained_variance_ratio_)
#plt.ylabel('% of variance explained')
#
#pca = decomposition.PCA(n_components=100) 
#pca.fit(train_data)
#pca.fit(test_data)
#PCtrain = pd.DataFrame(pca.transform(train_data))
#
#PCtest = pd.DataFrame(pca.transform(test_data))
#
#max(PCtrain.max())
#PCtrain=PCtrain/max(PCtrain.max())
#
#max(PCtest.max())
#PCtest=PCtest/max(PCtest.max())


#Import Libraries
from sklearn.model_selection import train_test_split
#----------------------------------------------------

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(train_data, train_lbl, test_size=0.25, random_state=123, shuffle =True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

#n_neig=np.arange(2,10)
#acc=[]
#for i in n_neig:
#    KNNModel = KNeighborsClassifier(n_neighbors= i)
#    KNNModel.fit(X_train[::10], y_train[::10])
#    acc.append(KNNModel.score(X_train, y_train))
#    
#plt.plot (n_neig,acc)
#plt.xlabel('neighbours')
#plt.xlim(2,20)
#plt.ylabel('accuracy')
#plt.show()

#as we 've seen at neighbours=4 best 
#import sklearn.externals.joblib as jb


model = KNeighborsClassifier(n_neighbors= 4)
model.fit(X_train, y_train)
#----------------------------------------------------


#----------------------------------------------------
#Applying GradientBoostingClassifier Model 

'''
'''

GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 
GBCModel.fit(X_train, y_train)
y_pred=GBCModel.predict(X_test)
#Calculating Details
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
#print('GBCModel features importances are : ' , GBCModel.feature_importances_)
#print('----------------------------------------------------')

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

#Calculating Prediction
#y_pred = GBCModel.predict(X_test)
#y_pred_prob = GBCModel.predict_proba(X_test)
#print('Predicted Value for GBCModel is : ' , y_pred[:10])
#print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])

#jb.dump(model , 'saved file.sav')
#
#
###############################################
#
#savedmodel = jb.load('saved file.sav')
#############################################
#----------------------------------------------------

#loading Voting Classifier
VotingClassifierModel = VotingClassifier(estimators=[('GBCModel',GBCModel),('KNNModel',model)], voting='hard',n_jobs=-1)
VotingClassifierModel.fit(X_train, y_train)

#Calculating Details
#print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
#print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))
#print('----------------------------------------------------')

#Calculating Prediction
y_pred = VotingClassifierModel.predict(X_test)
#print('Predicted Value for VotingClassifierModel is : ' , y_pred[:10])

a=pd.DataFrame(y_pred).iloc[y_pred==1 ]
#----------------------------------------------------
#Applying Cross Validate Score :  
'''
model_selection.cross_val_score(estimator,X,y=None,groups=None,scoring=None,cv=’warn’,n_jobs=None,verbose=0,
                                fit_params=None,pre_dispatch=‘2*n_jobs’,error_score=’raise-deprecating’)
'''

#  don't forget to define the model first !!!
#CrossValidateScoreTrain = cross_val_score(model, X_train, y_train, cv=5,n_jobs=-1)
#CrossValidateScoreTest = cross_val_score(model, X_test, y_test, cv=5,n_jobs=-1)
#
## Showing Results
#print('Cross Validate Score for Training Set: \n', CrossValidateScoreTrain)
#print('Cross Validate Score for Testing Set: \n', CrossValidateScoreTest)
#print('KNNClassifierModel Train Score is : ' , model.score(X_train, y_train))
#print('KNNClassifierModel Test Score is : ' , model.score(X_test, y_test))

y_pred = GBCModel.predict(test_data)

submission = pd.DataFrame(y_pred)
print(submission.shape)
submission.columns = ['Solution']
submission['Id'] = np.arange(1,submission.shape[0]+1)
submission = submission[['Id', 'Solution']]
submission.to_csv(r'd:/4.csv')

submission

