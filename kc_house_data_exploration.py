#!/usr/bin/env python
# coding: utf-8

# <h2 id="#sec1"> Importing Data & Exploration <h2>

# In[1]:


# General setup imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[136]:


# Regression analysis tools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from stepwise_selection import stepwise_selection
from sklearn.model_selection import cross_val_score


# In[3]:


kc_housing = pd.read_csv('kc_house_data.csv')
kc_housing.head()


# In[4]:


# Looking at columns type and if complete, and viewing which columns have null values that need to be addressed.
kc_housing.isna().sum(), kc_housing.info()


# In[5]:


# A general view of the shape of our columns data
_ = kc_housing.hist(figsize=(12,12))


# In[ ]:


# A very large (and slow) scatter matrix to look at all the potential relationships.


# In[ ]:


potential_cols = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                     'waterfront', 'view','condition','grade','sqft_above','sqft_basement',
                     'sqft_living15','sqft_lot15']
    
_ = pd.plotting.scatter_matrix(kc_housing[potential_cols], figsize=(12,12))


# <h2 id="sec2"> Scrubbing our data<h2>

# #### Notes on data cleaning and organizing:
# Dropping for now:
# * Id: Does not feel necessary for our exploration and will drop this column
# * Date: Going to use for some computations, but not do not need the variable on its own.
# * Sqft_above: With ample other features for our property size, we decided to pass on using the characteristic for the time being as it is so highly correlated to other features that feel can significantly contribute to our model, like sqft_living.
# * Latitude & Longitude: for location purses, we are going to do some exploring with the ZIP code.

# In[6]:


kc_measures = kc_housing.drop(['id','date','sqft_above','lat','long'], axis=1).copy()


# Cleaning NaN values:
# * Waterfront: data is both incomplete and appears to be very primarily negative. Since the value is already binary, we are marking all the NaN values negative as to match to majority. Since it is so skewed, we will re-evaluate its utitlity later.
# * Yr_renovated: To handle missing values, we collapsed this variable into binary for denoting renovation (True), or no value or no listed renovation year (False). Similar to waterfront, the majority did not have this value.
# * View: The view variable has a couple issues. First, there are some null values in this column.  Since there are only 63 and we have a significant amount of other rows to work with, we dropped these rows.  Second, the value for 'view' seems like a small difference between those with and without views, so it felt more useful to collapse the granuality of number of views to a simple binary for "viewed" (True), "not viewed"(False

# In[7]:


kc_measures['waterfront'] = kc_measures['waterfront'].fillna(0.0)

kc_measures['renovated'] = np.where(kc_measures['yr_renovated']>0, 1, 0) 
kc_measures.drop('yr_renovated', axis=1, inplace=True)

kc_measures['view'] = kc_measures['view'].dropna(axis=0)
kc_measures['view'] = np.where(kc_measures['view']> 0, 1, 0)


# Other data errors:
# * Sqft_basement: In exploring this column, we found that some were marked with a question mark instead of a value. It appears that this column was a computation from sqft_living. Our decision was to test this variable as a binary value equating to having a basement(True), or no basement listed (False)

# In[8]:


kc_measures['sqft_basement'] = kc_measures['sqft_basement'].str.replace('?', '0.0').astype('float')


# In[9]:


kc_measures['basement'] = np.where(kc_measures['sqft_basement'] > 0, 1, 0)
kc_measures.drop('sqft_basement', axis=1, inplace=True)


# Potential solutions for ZIP Code:
# * Creating dummy variables for each individual ZIP code. This may subdivide things too much, but can expirment with it. 
# * Trying a simple binary "In Seattle"-> True, "Not Seattle"-> False. A slimmer option, but might not provide the subltey of neighborhood variation.
# 
# In running tests of both, our ZIP codes as dummy variables appeared to more positively effect the metrics we were monitoring (R-Squared & RMSE), so we went with the dummy variables.  Below shows the process used for the "In Seattle" variation.

# In[10]:


df_zipcode_dums = pd.get_dummies(kc_measures['zipcode'])
kc_measures_wzip = pd.concat([kc_measures, df_zipcode_dums], axis=1)


# In[11]:


# seattle_zips = [98101, 98102, 98104, 98105, 98108, 98109, 98112, 98113, 98114, 98117, 98103, 98106, 98107,
#                 98111, 98115, 98116, 98118, 98119,98121, 98125, 98126, 98132, 98133, 98138, 98139, 98141, 
#                 98122, 98124, 98127, 98129, 98131, 98134, 98136, 98144, 98145, 98148, 98155, 98160, 98161, 
#                 98164, 98165, 98168, 98170, 98146, 98154, 98158, 98166, 98174, 98175, 98178, 98190, 98191,
#                 98177, 98181, 98185, 98188, 98189, 98194, 98195, 98199,98198]


# In[12]:


# kc_measures['in_seattle'] = kc_measures['zipcode'].map(lambda x: 1 if x in seattle_zips else 0)
# kc_measures.drop('zipcode', axis=1, inplace=True)


# Creating a Feature from Year Built:
# * Creating scalar values for the year value for "newness" of yr_built (absolute value of difference from 1899)

# In[13]:


kc_measures['yr_built_scalar_1899refyr'] = kc_measures['yr_built'].apply(lambda x: abs(1899-x))
kc_measures.drop('yr_built', axis=1, inplace=True)


# Further notes:
# * Bedrooms: The data appears complete, though has some heavy outliers on the higher end (one with 33 bedrooms!) which may need to be min-max standardized
# * Bathrooms: Comporable to bedrooms, may need to be adjusted for outliers, but complete.
# * Yr_built: To make this column more useful, we are going to convert it into its age in years (subtract current year)
# * Yr_renovated: data is fairly incomplete, so might want to skip
# * Sqft Living 15: Based on the graphs, it looks like we might find some helpful relationship
# 
# Here are the key features we ran our modelling tests on:

# In[14]:


kc_measures.columns


# <h2 id="sec3">Modeling<h2>

# ### Our first ugly child - A single linear regression model

# Notes on this regression:
# * We put both values on a standard scale just to see the effect and get an RMSE that might be more proportionally readble. 

# In[15]:


# Setting variables
X = kc_measures[['sqft_living']]
y = kc_measures[['price']]


# In[16]:


# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Calculate our y hat (how our model performs against the test data held off)
y_hat_test = linreg.predict(X_test)


# In[17]:


# See the R Squared score for on our test data
r2 = r2_score(y_test, y_hat_test)

# See our Squared Mean Error score for data
test_mse = mean_squared_error(y_test, y_hat_test)
test_rmse = np.sqrt(test_mse)

print(f"RMSE: {test_rmse}\nR2: {r2}")


# In[18]:


# Plotting
fig, ax = plt.subplots()
ax.scatter(X_test, y_test,  color='black')
ax.plot(X_test, y_hat_test, color='blue', linewidth=3)
ax.set_title('Sqft_living v. Price')
plt.show()


# ### Viola! 
# 
# What we learned from this test:
# * On its own, sqft_living does appear to be on the right track of variables to help predict price, based on its scoring and a brief assessment of the visualization.

# Potential mental breakdown of values as groups of characteristics:
# * Size - sqft_living, sqft_lot, basement, sqft_lot15, sqft_living15
# * Structure - bedrooms, bathrooms, floors, waterfront, view, condition, grade, renovated, basement
# * Location - zipcode

# ### Creating some multivariate regression models

# We made a quick function to help do the following for us:
# 1. Seperate out the Y ('price') and converting to log values (just for ease of reading) 
# 2. Setting our X (all other dataframe variables)
# 2. Create a train-test split on that data.
# 3. Run the training data through the linear regression function. 
# 4. Return a set of useful statistics to review their performance (R-Squared, MSE, RMSE, MAE)

# In[131]:


def lin_regress_summary(kc):

    y = kc['price']
    X = kc.drop(['price'], axis=1)

    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate our y hat (how our model performs against the test data held off)
    y_hat_test = model.predict(X_test)
    y_hat_test

    # See the R Squared score
    r2 = r2_score(y_test, y_hat_test)
    print(f"the R^2 score is: {r2}")

    # See our Mean Squared Error
    test_mse = mean_squared_error(y_test, y_hat_test)
    print(f"the test MSE is: {test_mse}")

    # See our Root Mean Squared Error
    test_rmse = np.sqrt(test_mse)
    print(f"the test RMSE is: {test_rmse}")
    
    # See our Mean Absolute Error
    test_mae = mean_absolute_error(y_test, y_hat_test)
    print(f"the test MAE is: {test_mae}")
    
    # Optional OLS test used to look into larger results, but ommitted for space and details:
    # model = sm.OLS(y, X)
    # results = model.fit()
    # print(results.summary())


# #### Testing all variables against price:

# In[20]:


lin_regress_summary(kc_measures)


# #### Testing all variables WITH all ZIP codes as dummy variables

# In[21]:


lin_regress_summary(kc_measures_wzip)


# Based just on this initial assessment, it appeared that the ZIP codes were a great help to improving our model, both in its representativeness to the data (R^2) and its potential for model prediction (RMSE & MAE).
# 
# With that knowledge, we began running several iterations of our variables without the ZIP codes to find the strongest ones, and then adding them along with our ZIP code variables.
# 
# First, a couple models based on that simple seperation of characteristics:

# In[22]:


kc_m_size = kc_measures[['price','sqft_living','sqft_lot',
                         'sqft_living15', 'sqft_lot15']]


# In[23]:


lin_regress_summary(kc_m_size)


# In[24]:


kc_m_structure = kc_measures[['price', 'bedrooms', 'bathrooms', 'floors', 
                              'waterfront', 'view', 'condition', 'grade', 
                              'renovated', 'basement']]


# In[25]:


lin_regress_summary(kc_m_structure)


# Next, we took a quick look at the results of a stepwise selector (function originally found on Learn.co, modified to take in a dataframe, know the 'price' is our key variable, and test the rest):

# In[34]:


stepwise_selection(kc_measures)


# Attempted as well with the ZIP codes variables, but in subsetting were marginally as useful as the complete set.

# In[92]:


# = pd.concat([kc_measures[['price']], df_zipcode_dums], axis=1)
# stepwise_selection(kc_zips_wprice)


# Based on these results, ran several more tests and tried a number of different configurations:

# In[35]:


kc_m_step_results = kc_measures[['price','waterfront','grade','yr_built_scalar_1899refyr', 'bathrooms',
                                 'sqft_living', 'bedrooms', 'view', 'sqft_lot15', 'condition',
                                 'floors', 'sqft_living15', 'renovated', 'basement']]


# In[36]:


lin_regress_summary(kc_m_step_results)


# In[37]:


kc_m_var_1 = kc_measures[['price', 'bedrooms', 'bathrooms', 'sqft_living',
                            'sqft_lot', 'view', 'condition', 'grade',
                            'zipcode', 'sqft_living15','sqft_lot15', 'renovated',
                            'basement', 'yr_built_scalar_1899refyr']]


# In[38]:


lin_regress_summary(kc_m_var_1)


# In[40]:


kc_m_var_2 = kc_measures[['price', 'bedrooms', 'bathrooms', 'sqft_living',
                            'sqft_lot', 'view', 'condition', 'grade', 'zipcode',
                            'sqft_living15','sqft_lot15',  'basement', 'yr_built_scalar_1899refyr']]


# In[41]:


lin_regress_summary(kc_m_var_2)


# In[42]:


kc_m_var_3 = kc_measures[['price', 'bedrooms', 'bathrooms', 'sqft_living',
                            'sqft_lot','view', 'condition', 'grade', 'sqft_living15',
                               'sqft_lot15','yr_built_scalar_1899refyr']]


# In[43]:


lin_regress_summary(kc_m_var_3)


# In[44]:


kc_m_var_4 = kc_measures[['price', 'bedrooms', 'bathrooms', 'sqft_living',
                            'sqft_lot', 'view',  'grade', 'sqft_living15',
                               'sqft_lot15', 'yr_built_scalar_1899refyr']]


# In[45]:


lin_regress_summary(kc_m_var_4)


# In[49]:


kc_m_var_5 = kc_measures[['price','waterfront', 'view', 'bedrooms',
                      'sqft_living', 'grade', 'yr_built_scalar_1899refyr']]


# In[50]:


lin_regress_summary(kc_m_var_5)


# With keeping in mind some of what we saw from our model tests, we ran further tests including our ZIP code dummy variables combined with our other variable measures:

# In[66]:


kc_use_6 = kc_measures[['price','waterfront', 'view', 'bedrooms',
                      'sqft_living', 'grade', 'yr_built_scalar_1899refyr']]
kc_m_var_6 = pd.concat([kc_use, df_zipcode_dums], axis=1)


# In[67]:


lin_regress_summary(kc_m_var_6)


# In[68]:


kc_use_7 = kc_measures[['price','sqft_living','bedrooms','bathrooms', 'grade', 'sqft_lot']]
kc_m_var_7 = pd.concat([kc_use, df_zipcode_dums], axis=1)


# In[69]:


lin_regress_summary(kc_m_var_7)


# Based on all our variations, it appears that our model for "kc_m_var_6" has the best potential.  From here, we will further check and test our model.

# <h2 id="sec4"> Multicollinearity testing and cross valuation<h2>

# We reviewed the ZIP codes as well, and did not find any individual values to be significantly effected by multicollinearity.
# 
# To check our other selected variables, we are looked for multicollinearity through the table below and explored the heatmap:

# In[95]:


kc_use_6.corr()


# In[94]:


#Heatmap view of how the variables relate:
corr = kc_use_6.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Based on this information, it appears that 'grade' is highly correlated with 'sqft_living', which does intrinsically make sense. To conteract that, we decided to drop that feature from our model.
# 
# Further, we tried to further look into our potential model features.

# In[141]:


kc_model = kc_measures[['price','waterfront','view','bedrooms','sqft_living', 'yr_built_scalar_1899refyr']].copy()

kc_model['bedrooms'] = np.log(kc_model['bedrooms'])

kc_model_complete = pd.concat([kc_model,df_zipcode_dums], axis=1)


# In[142]:


fig, ax = plt.subplots(1,2, figsize=(12,5))

ax[0].scatter(kc_model['sqft_living'], kc_model['price'])
ax[0].set_title("Sqft_living vs. price")

ax[1].scatter(kc_model['bedrooms'], kc_model['price'])
ax[1].set_title("Bedrooms vs. price")

plt.show()


# In[143]:


lin_regress_summary(kc_model_complete)


# #### Cross validation with K-Folds

# In[164]:


y = kc_model_complete['price']
X = kc_model_complete.drop('price', axis=1)

linreg = LinearRegression()

cv_10_results = np.mean(cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error"))
cv_20_results = np.mean(cross_val_score(linreg, X, y, cv=20, scoring="neg_mean_squared_error"))


# <h2 id="#sec5">Graveyard of Calculated Features<h2>

# The following are a seleciton of features we drafted, modeled with other variables, and found not a significant enough impact to include in our model.

# #### Yard Size: all square footage outside of the living area (Sqft_Living - Sqft_Lot)

# In[59]:


# kc_measures['yard'] = kc_measures['sqft_living'] - kc_measures['sqft_lot']


# #### Seasons: taking the month sold during the year and grouping into four seasons. Tried just months as well, but sales were fairly even by month.

# In[56]:


# kc_mod['month'] = kc_mod['date'].apply(lambda x: x[:2]).str.replace('/','').astype(int)

# def quarter(val):
#     if val in range(3):
#         return '1-3'
#     elif val in range(4,7):
#         return '4-6'
#     elif val in range(7,10):
#         return '7-9'
#     elif val in range(10,13):
#         return '10-12'
    
# kc_mod['month'] = kc_mod['month'].apply(lambda x: quarter(x))
# kc_mod['month'].value_counts()
# df_month_dums = pd.get_dummies(kc_mod['month'])
# kc_mod = pd.concat([kc_mod, df_month_dums], axis=1)


# #### Distance from Expensive neighborhoods: Based on article found from 2017 with most expensive neighborhoods, create values for distance from those expensive neighborhood centers.  This method we did not finish testing, but feel that some of its subtley may be picked up in ZIP codes 

# <a href="https://seattle.curbed.com/2017/10/11/16462132/seattle-cheap-expensive-neighborhoods-buying-home">Original article<a>

# In[58]:


# Test for location close to expensive areas
# kc_measures['loc'] = (kc_measures['lat'] + 90) * (180 + kc_measures['long'])
# kc_measures.drop(['lat','long'], axis=1, inplace=True)
# # location = (lat + 90) * 180 + long

# downtown = (47.60806 + 90) * (180 + -122.33611)
# madrona = (47.613274 + 90) * (180 + -122.28887)
# slu = (47.62343 + 90) * (180 + -122.33435)
# eastlake = (47.64708 + 90) * ( 180 + -122.32477)
# queen_anne = (47.63692 + 90) * (180 + -122.35579)
# magnolia = (47.65056 + 90) * (180 + -122.40083)
# first_hill = (47.60864 + 90) *(180 + -122.32679)

# #location = (lat + 90) * 180 + long

# kc_expensive = kc_housing.drop(['id','date','sqft_above'], axis=1).copy()
# kc_expensive['loc'] = (kc_measures['lat'] + 90) * (180 + kc_measures['long'])
# kc_expensive['loc_downtown'] = kc_expensive['loc'].apply(lambda x: x - downtown)

# kc_expensive['loc_madrona'] = kc_expensive['loc'].apply(lambda x: x - madrona)
# kc_expensive['loc_slu'] = kc_expensive['loc'].apply(lambda x: x - slu)
# kc_expensive['loc_eastlake'] = kc_expensive['loc'].apply(lambda x: x - eastlake)
# kc_expensive['loc_queen_anne'] = kc_expensive['loc'].apply(lambda x: x - queen_anne)
# kc_expensive['loc_magnolia'] = kc_expensive['loc'].apply(lambda x: x - magnolia)
# kc_expensive['loc_first_hill'] = kc_expensive['loc'].apply(lambda x: x - first_hill)

