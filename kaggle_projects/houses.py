import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split


##################################
data = pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')

# =============================================================================

train_df=data.drop(['Id','PoolQC','MiscFeature','Alley','Fence','TotRmsAbvGrd'
                    ,'GarageYrBlt'],axis=1)
test_data=data_test.drop(['Id','PoolQC','MiscFeature','Alley','Fence','TotRmsAbvGrd'
                    ,'GarageYrBlt'],axis=1)
# =============================================================================
train_df=pd.DataFrame(train_df)
test_data.info()
# =============================================================================
#missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


total_ = test_data.isnull().sum().sort_values(ascending=False)
percent_ = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_, percent_], axis=1, keys=['Total', 'Percent'])

# =============================================================================
plt .scatter(train_df['YearBuilt'],train_df['SalePrice'])
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')
plt.show()
# =============================================================================
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Note that there is a strong relationship between Garage cars &area && totrm--&grlive && TotalBs &1st
#=====================================================

#Missing data 
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)

train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)

test_data = test_data.drop((missing_data[missing_data['Total'] > 1]).index,1)


#===================================================
from sklearn.preprocessing import LabelEncoder
cols = (  
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 
         'Functional', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street',  'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold','MSZoning','LandContour','Utilities','LotConfig','Neighborhood','Condition1','Condition2','BldgType',
        'HouseStyle','RoofStyle','RoofMatl','Exterior2nd','Foundation','Heating','Electrical','SaleType','SaleCondition','Exterior1st')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_df[c].values)) 
    train_df[c] = lbl.transform(list(train_df[c].values))
    lbl.fit(list(test_data[c].values)) 
    test_data[c] = lbl.transform(list(test_data[c].values))


#===================================================
   #dropping laiers 
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    
train_df=train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<200000)] .index)
#===================================================
pred_train=train_df['SalePrice']
train_df=train_df.drop(['SalePrice'],axis=1)

from sklearn.impute import SimpleImputer

ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(test_data)
test_data = ImputedX.transform(test_data)

#=====================================
#standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
train_df = scaler.fit_transform(train_df)
test_data = scaler.fit_transform(test_data)
#================================================
#train _test _splitting
X_train, X_test, y_train, y_test = train_test_split(train_df, pred_train, test_size=0.25, shuffle =True,random_state=40)

#================================================

#Modeling
from sklearn.ensemble import RandomForestRegressor


RandomForestRegressorModel = RandomForestRegressor(n_estimators=240,max_depth=20, random_state=33, n_jobs=-1)
RandomForestRegressorModel.fit(X_train, y_train)

#Calculating Details
print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))


#Calculating Prediction
y_pred = RandomForestRegressorModel.predict(test_data)
d=pd.DataFrame(y_pred)
d.to_csv('D:\\2.csv')

#----------------------------------------------------

from sklearn.model_selection import GridSearchCV


#=======================================================================
a=np.arange(100,400,20)
SelectedParameters = {'n_estimators':[i for i in a], 'max_depth':[j for j in range (10,30)]}
#=======================================================================
GridSearchModel = GridSearchCV(RandomForestRegressorModel,SelectedParameters, cv = 2,return_train_score=True)
GridSearchModel.fit(X_train, y_train)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)



#=================================================
#accuracy
from sklearn.metrics import mean_squared_error

 

MSEValue = mean_squared_error(y_test, y_pred, multioutput='raw_values') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)








