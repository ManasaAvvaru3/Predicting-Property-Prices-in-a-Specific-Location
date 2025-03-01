#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv('C:\\Users\\91797\\Downloads\\Property\\train.csv')
test = pd.read_csv('C:\\Users\\91797\\Downloads\\Property\\test.csv')


# In[3]:


print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)


# In[4]:


train.head()


# In[5]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[6]:


train.SalePrice.describe()


# In[7]:


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


# In[8]:


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# In[9]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[10]:


corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])


# In[11]:


train.OverallQual.unique()


# In[12]:


quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)


# In[13]:


quality_pivot


# In[16]:


quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)


# In[17]:


plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()


# In[18]:


plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[19]:


train = train[train['GarageArea'] < 1200]


# In[20]:


plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[21]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls


# In[22]:


print ("Unique values are:", train.MiscFeature.unique())


# In[26]:


features = {
 'Elev': 'Elevator',
 'Gar2': '2nd Garage (if not described in garage section)',
 'Othr': 'Other',
 'Shed': 'Shed (over 100 SF)',
 'TenC': 'Tennis Court',
 'NA': 'None'
}


# In[27]:


categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()


# In[28]:


print ("Original: \n") 
print (train.Street.value_counts(), "\n")


# In[29]:


train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)


# In[30]:


print ('Encoded: \n') 
print (train.enc_street.value_counts())


# In[31]:


condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[32]:


def encode(x):
 return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# In[33]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[34]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# In[35]:


sum(data.isnull().sum() != 0)


# In[36]:


y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


# In[38]:


from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[39]:


model = lr.fit(X_train, y_train)


# In[40]:


print ("R^2 is: \n", model.score(X_test, y_test))


# In[41]:


predictions = model.predict(X_test)


# In[42]:


from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[43]:


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,color='b') 

#alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[46]:


for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test, y_test),mean_squared_error(y_test, preds_ridge))
    plt.annotate(overlay, xy=(12.1, 10.6), fontsize='x-large')
    plt.show()


# In[47]:


sublesson = pd.DataFrame()
sublesson['Id'] = test.Id


# In[48]:


feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()


# In[49]:


predictions = model.predict(feats)


# In[50]:


final_predictions = np.exp(predictions)


# In[51]:


print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])


# In[52]:


sublesson['SalePrice'] = final_predictions
sublesson.head()


# In[54]:


sublesson.to_csv('Final_Sub.csv', index=False)


# In[ ]:




