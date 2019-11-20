#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
sns.set(color_codes=True)

from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, LinearRegression, RidgeCV, Lasso
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error, r2_score, mean_squared_error,accuracy_score

from sklearn import metrics
from sklearn import linear_model
from matplotlib import pyplot
from xgboost import plot_importance, XGBClassifier
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor


# In[4]:


data = pd.read_csv('C:/Users/Dell_pc/Risky Dealer Case Study Transactions.csv')
#print(pd.isnull(data).sum())


# In[5]:


#Drop null values from JDPowersCat
data.dropna(subset=['JDPowersCat'], how='all', inplace=True)
data.dropna(subset=['CarMake'], how='all', inplace=True)

#Imputer cols with mean values
cols = ['Autocheck_score']
for col in cols:
    data[col].fillna(data[col].mean(), inplace=True)
    
#Replace char with numeric values
data['ConditionReport'] = data['ConditionReport'].str.replace('SL','0')
data['ConditionReport'] = data['ConditionReport'].str.replace('PR','10')
data['ConditionReport'] = data['ConditionReport'].str.replace('RG','20')
data['ConditionReport'] = data['ConditionReport'].str.replace('AV','30')
data['ConditionReport'] = data['ConditionReport'].str.replace('CL','40')
data['ConditionReport'] = data['ConditionReport'].str.replace('EC','50')
    
data['ConditionReport'] = pd.to_numeric(data['ConditionReport'], errors='coerce')

#LabelEncode categorical values
data['SellingLocation'] = LabelEncoder().fit_transform(data['SellingLocation'])
data['CarMake'] = LabelEncoder().fit_transform(data['CarMake'])
data['JDPowersCat'] = LabelEncoder().fit_transform(data['JDPowersCat'])


# In[6]:


features = ['Mileage', 'SellingLocation', 'CarMake', 'SalePrice', 'CarYear','JDPowersCat','MMR','LIGHTG',
            'LIGHTY','LIGHTR','DSEligible','Arbitrated','Salvage','Autocheck_score','PSIEligible','Returned']
data1 = data[features]

#Drop values with null in the target variable
data1 = data1.dropna()
#data1 = data1.reset_index(drop=True)
#print(data1.head())

#Values with null in the target variable as test data
test = data[data['Returned'].isnull()]
#print(test.head())


# In[7]:


#data.info()
Returned_col = data.iloc[:,-1]
data_NR = data.drop(labels='Returned', axis=1)
#data_NR.info()


# In[ ]:


#To predict values for condition report

data_CR = data_NR.dropna()
test_CR = data_NR[data_NR['ConditionReport'].isnull()]

#RandomForestRegressor to predict values
feat_CR = ['Mileage', 'SellingLocation', 'CarMake', 'SalePrice', 'CarYear','JDPowersCat','MMR','LIGHTG',
            'LIGHTY','LIGHTR','DSEligible','Arbitrated','Salvage','Autocheck_score','PSI','PSIEligible']
X_cr = data_CR[feat_CR]
y_cr = data_CR.ConditionReport


# In[ ]:


#Random Forest Regressor to predict Condition Report values

X_train_cr, X_val_cr, y_train_cr, y_val_cr = train_test_split(X_cr, y_cr, test_size=0.3, random_state=1)
lr = RandomForestRegressor(n_estimators=100)

lr.fit(X_train_cr, y_train_cr)
y_hat = lr.predict(X_val_cr)      #this gives me my predictions
print('Accuracy of RandomForestRegressor:',lr.score(X_val_cr, y_val_cr))

r2 = r2_score(y_val_cr, y_hat)
print(r2)


# In[ ]:


#Hypothesis Testing between categorical variables

from scipy.stats import chi2_contingency
from scipy import stats

chi = ['CarYear', 'CarMake','JDPowersCat','SellingLocation','SalePrice']
df_chi = data_CR[chi]

contingency_table=pd.crosstab(df_chi["CarYear"], df_chi["SalePrice"])
#print('contingency_table :-\n',contingency_table)
#Observed Values
Observed_Values = contingency_table.values 
#print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
#print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)
#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[ ]:


from scipy.stats import pearsonr
data1, data2 = [X_cr['SalePrice'], X_cr['MMR']]
corr, p = pearsonr(data1, data2)

print(corr)


# In[ ]:


X_test = test_CR[feat_CR]
y_final = lr.predict(X_test)


# In[ ]:


values = []
for each in y_final:
    values.append(int(each))
#print(values)

#Replacing null values with predicted CR values 
new_data_1 = test_CR.assign(ConditionReport = values)
new_data_1.head()


# In[ ]:


#Concat the predicted rows in original data

frames_1 = [new_data_1,data_CR]
df_1 = pd.concat(frames_1, axis=0, join = 'outer', ignore_index=False, keys=None,
         levels=None, names=None, verify_integrity=False, copy=True)
df_1 = df_1.sort_index(axis = 0)

#Adding Returned column to the data
df = df_1.assign(Returned = Returned_col)

#print(pd.isnull(df).sum())


# In[ ]:


# Feature importance using XGB Classifier

model = XGBClassifier()
model.fit(X_train_cr, y_train_cr)
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
#
plot_importance(model)
pyplot.show()


# In[ ]:


#Testing Different models for prediction

xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb_model.fit(X_train_cr, y_train_cr)
y_pred = xgb_model.predict(X_val_cr)
mse=mean_squared_error(y_val_cr, y_pred)
print(np.sqrt(mse))
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_val_cr, predictions)
print('Accuracy of XGB model to predict Condition Report values :',accuracy)

#Linear Regression 

lr = LinearRegression()
scores = cross_val_score(lr, X_train_cr, y_train_cr, cv = 5)
predictions = cross_val_predict(lr, X_val_cr, y_val_cr, cv = 5) 
accuracy = metrics.r2_score(y_val_cr, predictions)
#print(accuracy)

lr.fit(X_train_cr, y_train_cr)
y_hat = lr.predict(X_val_cr) #this gives me my predictions
print('Accuracy of Linear Regression CV model to predict Condition Report values :',lr.score(X_val_cr, y_val_cr))


# In[ ]:


#separating data to check null values in the test set
data1 = df.dropna()
test_Ret = df[df['Returned'].isnull()]

#test_Ret.head()


# In[ ]:


feat = ['Mileage', 'SellingLocation', 'CarMake', 'SalePrice', 'CarYear','JDPowersCat','MMR','LIGHTG',
            'LIGHTY','LIGHTR','DSEligible','Arbitrated','Salvage','Autocheck_score','PSIEligible']
X = data1[feat]
y = data1.Returned


#X_train, X_test, y_train, y_test = train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
#multi_class='multinomial',
logReg = LogisticRegression(solver='lbfgs',random_state=0,max_iter = 3000,multi_class='multinomial')
model = logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_val)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logReg.score(X_val, y_val)))



#Logistic Regression with cross validation

logRegCV = LogisticRegressionCV(cv=5, random_state=0,max_iter = 5000).fit(X, y)
model_CV = logRegCV.fit(X_train, y_train)
y_pred = logRegCV.predict(X_val)
print('Accuracy of CV logistic regression classifier on test set: {:.2f}'.format(logRegCV.score(X_val, y_val)))
score =accuracy_score(y_val,y_pred)
print(score)


#Lasso Regression

lasso = Lasso(alpha=0.00001, max_iter=10e5)
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
Val_score=lasso.score(X_val,y_val)
coeff_used = np.sum(lasso.coef_!=0)
print('Accuracy of Lasso regression classifier on test set: {:.2f}'.format(lasso.score(X_val, y_val)))
print ("number of features used: ",coeff_used)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)

clf.fit(X_train, y_train)
y_pred_2 = clf.predict(X_val)

print('Accuracy of CV logistic regression classifier on test set: {:.2f}'.format(clf.score(X_val, y_val)))


from sklearn.metrics import accuracy_score
score_2 =accuracy_score(y_val,y_pred_2)
print('Accuracy of Random Forest Classifier classifier on test set:',score_2)


# In[ ]:


#Predict Returned values

X_test = test_Ret[feat]
y_pred_ret = clf.predict(X_test)


# In[ ]:


#Roc curve to test accuracy and overfitting

logit_roc_auc = roc_auc_score(y_val, clf.predict(X_val))
fpr, tpr, thresholds = roc_curve(y_val, clf.predict_proba(X_val)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:


#replacing null values with predicted values
new_data_2 = test_Ret.assign(Returned = y_pred_ret)
new_data_2.head()


#Concat test values to org dataframe

frames_2 = [new_data_2,data1]
final_df = pd.concat(frames_2, axis=0, join = 'outer', ignore_index=False, keys=None,
         levels=None, names=None, verify_integrity=False, copy=True)
final_df = final_df.sort_index(axis = 0)


# In[ ]:


final_df.to_csv(r'C:\Users\Dell_pc\Risky Dealer Case Study Transactions_final.csv', index = None, header=True)

