#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (ie 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - 
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[1]:


import pandas as pd
import numpy as np
import math
import json
from clean_data import convert_to_datetime, clean_profile_data, clean_portfolio_data, clean_transcript_data
from clean_data import create_offer_analysis_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from time import time
from visuals import plot_gender_income_distribution, plot_gender_age_distribution, feature_plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# <h3>Possible Questions</h3>

# <ul>
#     <li>
#         How likely is someone to use the offer based on when they signed up for the app? </li>
#         <li>Which offers to send women for certain age range?</li>
#         <li>which offers to send men for certain age range?</li>
#         <li>which offers should be sent based on income/gender?</li>
#     
#     
# </ul>

# <h3>1. Explore the Data</h3>

# In[2]:


pd.set_option('display.max_rows', 1000)
pd.set_option.float_format = '{:.2f}'.format
pd.set_option('display.max_colwidth', -1)


# <h4>1.1 portfolio EDA</h4>

# In[3]:


portfolio.head()


# check for missing values

# In[4]:


portfolio.isnull().sum()


# There are no missing values in the portfolio dataframe

# In[5]:


portfolio.describe()


# <h4>1.2 profile EDA</h4>

# In[6]:


profile.head()


# In[7]:


profile.info()


# <h4>1.2.1 check for missing values</h4>

# In[8]:


profile.isnull().sum()


# Seems to be 2,175 missing values in the gender and income columns

# In[9]:


profile.describe()


# <h4>1.2.2 View age distribution as there seems to be some entries which are entered as 118</h4>

# In[10]:


profile.age.hist();


# There appears to be a number of age entries which do not make sense. So might be able to drop them.

# In[11]:


print(profile[profile['age'] > 110].count())


# In[12]:


profile[['gender', 'income', 'age']][profile['age'] > 110].head()


# It appears that age 118 corresponds to the null values for the gender and income column so can drop along with the NaNs in the data cleaning steps

# <h4>1.2.3 What is the gender distribution?</h4>

# In[13]:


profile['gender'].value_counts()


# In[14]:


profile['gender'].value_counts().plot(kind="bar");


# The gender distribution is as follows: 8,484 men, 6,129 women, and 212 other 

# <h4>1.2.4 Explore income distribution</h4>

# In[15]:


#plot histogram
profile['income'].hist();


# It seems that the most commone income range is between 50,000 and 70,000 dollars

# <h4>1.2.4 Look at when the customer signed up for the offers</h4>

# In[16]:


profile.became_member_on.head()


# Determine range of years customers signed up for the app

# In[17]:


date_joined = profile['became_member_on'].apply(convert_to_datetime)

#determine year
start_year = date_joined.apply(lambda elem: elem.year).value_counts()
start_year


# In[18]:


ax = start_year.plot.bar(title='Year Customers Signed up for the App')


# It looks like most customers signed up for the app in 2017

# Determine range of months when customers signed up for app

# In[19]:


start_month = date_joined.apply(lambda elem: elem.month).value_counts()

start_month *= 100 / start_month.sum()
start_month


# In[20]:


ax = start_month.plot.bar(title='Month Customers Signed up for the App')


# October was the most common month for customers to create an account; however, there does not seem to be a large variation between the months. Based on the results, it doesn't seem like the time of year has an impact on when people signed up for the app

# <h4>1.3 transcript EDA</h4>

# In[21]:


transcript.head()


# In[22]:


transcript.info()


# <h4>1.3.1 Find number of missing values</h4>

# In[23]:


transcript.isnull().sum()


# There are no missing values in the transcript dataframe

# <h4>1.3.2 Determime number of different event types.</h4> 

# In[24]:


total_events = transcript['event'].value_counts()


# Find percent of each event action

# In[25]:


#convert value counts results to dataframe to calculate percents
total_events = pd.DataFrame(list(zip(total_events.index.values, total_events)), columns = ['event', 'count'])
total_events


# In[26]:


#calculate transaction percent
percent_transaction = (total_events.iloc[0]['count'] / total_events['count'].sum()) * 100
print('The percent transactions out of the total events is:', percent_transaction, '%')

#calculate percent offer received
percent_offer_received = (total_events.iloc[1]['count'] / total_events['count'].sum()) * 100
print('The percent received offers out of the total events is:', percent_offer_received, '%')

#calculate percent offer viewed
percent_offer_viewed = (total_events.iloc[2]['count'] / total_events['count'].sum()) * 100
print('The percent viewed offers out of the total events is:', percent_offer_viewed, '%')

#calculate percent offer completed
percent_offer_completed = (total_events.iloc[3]['count'] / total_events['count'].sum()) * 100
print('The percent completed offers out of the total events is:', percent_offer_completed, '%')


# <h3>2. Clean Data</h3>

# <h4>2.1 Clean portfolio dataframe</h4>

# In[27]:


#Use clean_portfolio_data function from clean_data helper file to return cleaned portfolio dataframe
portfolio = clean_portfolio_data(portfolio)


# In[28]:


portfolio.head()


# <h4>2.2 Clean profile dataframe</h4>

# In[29]:


#Use clean_profile_data function from clean_data helper file to return cleaned profile and a separate age and 
#gender dataframe
profile, age_gender_income_df = clean_profile_data(profile)


# In[30]:


profile.head()


# In[31]:


age_gender_income_df.head()


# <h4>2.3 Clean transcript dataframe</h4>

# In[32]:


#Use clean_transcript_data function from clean_data helper file to return cleaned offer_df and 
#transaction_df dataframes
offer_df, transaction_df = clean_transcript_data(transcript, profile)


# In[33]:


offer_df.head()


# In[34]:


transaction_df.head()


# In[35]:


offer_df.info()


# In[36]:


transaction_df.info()


# <h4>2.4 Combine portfolio, profile, offer_df, and transaction_df dataframes</h4>

# In[37]:


#Use create_offer_analysis_dataset function from clean_data helper file to return a dataframe which combines
#the profile data, portfolio data, offer_df, and transaction_df data.
#Note that this step takes several minutes to complete.
combined_df = create_offer_analysis_dataset(profile, portfolio, offer_df, transaction_df)


# In[38]:


#move offer_id and total_amounts columss to front of dataframe to make easier to work with dataframe
column_ordering = ['offer_id', 'total_amount']

#move offer_id and total_amount columns to beginning of dataframe
column_ordering.extend([i for i in combined_df.columns if i not in column_ordering])

combined_df = combined_df[column_ordering]
combined_df.head()


# In[39]:


combined_df.head()


# <h3>3. Statistics </h3>

# <h4>3.1 Profile DataFrame Stats </h4>

# <h4>3.1.1 General income stats</h4>

# In[40]:


#get stats for income column
profile[profile['income'].notnull()].describe()


# <h4>3.1.2 Income distribution for Men and Women </h4>

# In[41]:


#use plot_gender_income_distribution function from visuals helper file
plot_gender_income_distribution(profile)


# The men tended to have a lower income distribution with a peak around 60,000 dollars than the females with a peak around 80,000 dollars who used the app

# <h4>3.1.3 Income distribution by Age </h4>

# In[42]:


#plot income vs age as scatter plot.
g = sns.pairplot(age_gender_income_df[['age','income']],diag_kind="kde", height=5, aspect=2)


# Income and age don't appear to have very good relation based on the above plots.

# <h4>3.1.3 When did people tend to join? </h4>

# In[43]:


profile[profile['membership_start_year'].notnull()].describe()


# In[44]:


profile[profile['membership_start_year'].notnull()].info()


# In[45]:


sns.distplot(profile['membership_start_year']).set_title("Distribution of Membership Year Joined");


# It seems that most customers joined the app in 2017

# <h4>3.1.4 Customers age range </h4>

# In[46]:


#use plot_gender_age_distribution function from visuals helper file
plot_gender_age_distribution(age_gender_income_df)


# On average it appears that more men between ages 20 to 40 use the app, while more women 40 and older seem to use the app then men of similiar age

# <h4>3.2 Transcript and offer Data Stats</h4>

# In[47]:


transcript.describe()


# In[48]:


offer_df.describe()


# In[49]:


transaction_df.describe()


# <h4>3.2.1 Relationship between offer length and offer completion </h4>

# In[50]:


#plot days vs completed scatter plot.
g = sns.pairplot(offer_df[['days','completed']],diag_kind="kde", height=5, aspect=2)


# From the plots above, 

# <h4>3.2.2 Relationship between offer length and offer amount </h4>

# In[51]:


#plot days vs amount.
g = sns.pairplot(transaction_df[['days','amount']],diag_kind="kde", height=5, aspect=2)


# <h4>3.3 portfolio data stats</h4>

# In[52]:


portfolio.describe()


# <h4>3.3.1 Cleaned Portfolio DataFrame Exploration Plots</h4>

# In[53]:


#Plot days_duration vs difficulty
bp = sns.barplot(x="days_duration",y="difficulty",data = portfolio, ci=None).set_title("Offer Difficulty vs Days Duration")
axes = bp.axes
axes.set_ylim(0,20)
plt.show()


# In[54]:


#Plot days_duration vs reward
bp = sns.barplot(x="days_duration",y="reward",data = portfolio, ci=None).set_title("Days Duration vs Reward");
axes = bp.axes
axes.set_ylim(0,)
plt.show()


#  <h3>4. Prediction Models</h3>

# <h4> 4.1 Set feature to be predicted </h4>

# In[55]:


#feature to be predicted
target_name = 'completed'

#features to train the model
variables = combined_df.drop(columns=[target_name, 'offer_id'])

#copy target dataframe for predicition
target = combined_df.filter([target_name])


# <h4> 4.2 Split the data into training and testing </h4>
# 

# In[56]:


# Split the 'features' and 'income' data into training and testing sets
random_state = np.random.RandomState(0)
        
X_train, X_test, y_train, y_test = train_test_split(variables, target,
                                                    test_size = 0.2, 
                                                    random_state = random_state)

variable_names = variables.columns[2:]

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[57]:


X_train.head()


#  <h4>4.3 Perform Naive Bayes predictor</h4>

# In[58]:


#naive predictor
naive_predictor_accuracy = accuracy_score(y_train, np.ones(len(y_train)))
naive_predictor_f1score = f1_score(y_train, np.ones(len(y_train)))

print('Naive predictor accuracy: %.3f' % (naive_predictor_accuracy))
print('Naive predictor f1-score: %.3f' % (naive_predictor_f1score))


# <h4>4.4 Model Tuning </h4>

# In[59]:


#Initialize the classifier
clf = RandomForestClassifier(random_state = 40, n_estimators=100)

#Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10], 'min_samples_split':[2,4,6,8,10]}

#Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(f1_score)

#Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV
grid_obj = GridSearchCV(clf, parameters, cv=5, scoring=scorer)

#Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

#get the best estimator
best_clf = grid_fit.best_estimator_

#make predictions using the unoptimized model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

#report before and after scores
print('Unoptimized model\n-----')
print('Accuracy score on testing data:{:.4f}'.format(accuracy_score(y_test, predictions)))
print('F-score on testing data:{:.4f}'.format(fbeta_score(y_test, predictions, beta=0.5)))
print('\nOptimized model\n-----')
print('Final accuracy score on testing data:{:.4f}'.format(accuracy_score(y_test, best_predictions)))
print('Final F-score on testing data:{:.4f}'.format(fbeta_score(y_test, best_predictions, beta=0.5)))


# <ul>
# <li>The unoptimized model has an Accuracy Score on the testing data of 0.9048 and an F-score of 0.8939</li>
# <li>The optimized model has an Accuracy Score on the testing data of 0.9124 and an F-score of 0.8942</li>
# </ul>
# Optimizing the model using GridSearchCV improves the accuracy and F-score of the model.

# <h4>4.5 Feature Importance</h4>

# In[60]:


#Import Random Forest Classifier model that has 'feature_importances_'
clf = RandomForestClassifier(random_state = 600)

#Train the supervised model on the training set using .fit(X_train, y_train)
model = clf.fit(X_train, y_train)

#Build important features dictionary from Random Forest Classifier
important_features_dict = {}
for x,i in enumerate(clf.feature_importances_):
    important_features_dict[x]=i

#Save features dictionary to dataframe
feature_importance = pd.DataFrame(list(zip(variable_names, important_features_dict)),
                 columns=['feature', 'importance'])

#sort feature_importance dataframe by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

print('Most important features: %s' %feature_importance)


# <h4>4.5.1 Plot Random Forest Feature Importance</h4>

# In[61]:


#Plot features using feature_plot function from visuals helper file
feature_plot(feature_importance)


# <h4>4.6 LightGBM</h4>

# In[62]:


# create dataset for lightgbm from training and testing data
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[63]:


#tune parameters
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
#params['metric'] = ('l1', 'l2')
params['sub_feature'] = 0.5
params['num_leaves'] = 50
params['min_data'] = 500
params['max_depth'] = 100


# In[64]:


print('Starting training...')
# train
evals_result = {}  # to record eval results for plotting
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                feature_name = list(X_train),
                evals_result=evals_result,
                early_stopping_rounds=5,
                verbose_eval=20)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')


# In[65]:


print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#convert to binary values
for i in range(len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0


# <h4>4.6.1 Plot metrics</h4>

# In[66]:


print('Plotting metrics recorded during training...')
ax = lgb.plot_metric(evals_result, metric='binary_logloss')
plt.show()


# <h4>4.6.2 Plot LightGBM Feature Importance</h4>

# In[71]:


print('Plotting feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=20, figsize=(20,10), title='LightGBM Feature Importance')
plt.show()


# <h4>4.6.3 Create confusion matrix and measure accuracy of LightGBM model</h4>

# In[68]:


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', cm)

#Accuracy
accuracy=accuracy_score(y_pred,y_test)
print('LightGBM accuracy:', accuracy)


# <h3>Conclusion</h3>

# The problem I chose to solve was to determine whether or not a customer will respond to an offer through the Starbucks app. I approached the problem using two methods, determining the most influential features and implementing a predictive model which can be used to determine if a customer will complete and offer.
# I accomplished this task through the following steps:
# <ul>
#     <li>Exploring the data </li>
#     <li>Cleaning the data</li>
#     <li>Explore Data Statistics</li>
#     <li>Implement Prediction Models</li>
#     
# </ul>
# 
# Through exploring the data and cleaning the data, the three data files provided (portfolio, profile and transcript) were combined into one dataframe in order to split into training at testing sets. 
# The data statistics section provided insight into the relationship between features such as gender and income, age and income, and length of offer and offer reward. 
# 
# The top features according to the Random Forest Model are:
# <ul>
#     <li>Offer reward</li>
#     <li>Membership start year </li>
#     <li>Informational type of offer </li>
#     <li>Customer's income </li>
#     <li>Discount offer type </li>
# </ul>
# 
# The top features according to the LightGBM Model are:
# <ul>
#     <li>Total amount spend by customer</li>
#     <li>Customer's income </li>
#     <li>Offer difficulty </li>
#     <li>Length of offer </li>
#     <li>Offer reward </li>
# </ul>
# 
# 
# The best model prediction method was chosen by comparing the accuracy and F-score of a Naive Bayes model, Random Forest, and the accuracy of LightBGM model. 
# <ul>
#     <li>Naive predictor accuracy: 0.471
#         Naive predictor f1-score: 0.640 </li>
#     <li> Optmizied Random Forest accuracy: 0.9124
# f1-score: 0.8943</li>
#     <li> LightGBM accuracy: 0.9114</li>
# </ul>
# 
# The Random Forest model had the highest accuracy of 0.9124 while the LightGBM model had an accuracy score of 0.9114. However, the LightGBM model had a signficantly faster computation time. To achieve such a high accuracy and F-score for the Random Forest model, the model had to optimized using the GridSearchCV algorithim, adding approximately an additional 20 minutes of computation time while the LightGBM model had a much faster computation time. Since the accuracy scores are so close, the LightGBM model seems to be the better model for prediction in this case.
# 

# In[ ]:




