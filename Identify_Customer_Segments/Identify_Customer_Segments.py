#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import pprint

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')


# In[3]:


azdias.shape


# In[4]:


t = azdias['CAMEO_DEU_2015']
sum(pd.isnull(t))


# In[5]:


azdias['ALTERSKATEGORIE_GROB'].value_counts()


# In[6]:


pd.set_option('display.max_columns',100)
azdias.describe(include='all')


# In[7]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

azdias.head(n=100)


# In[8]:


feat_info


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[9]:


#get total number of element
num_of_data = np.prod(azdias.shape)


# In[10]:


#get percent of naturally missing elements in azdias
natural_miss_count = 0 #initialize counter

for i in azdias.count():
    natural_miss_count += i
    
percent_miss = 100*(1-natural_miss_count/num_of_data)

print("The percentage of naturally missing values is:", " ", percent_miss,"%", sep='')


# In[11]:


for col in range(azdias.shape[1]):  #loop through azdias columns
    column_name = azdias.columns[col]  #get column name
    missing_list = feat_info.iloc[col,3]  #get missing_or_unknown column from data_dictionary
    missing_list = missing_list.replace('[','') #remove left bracket from string
    missing_list = missing_list.replace(']','') #remove right bracket from string
    missing_list = missing_list.split(',')  #split into individual strings
    
    #find data that is natually missing and continue loop to omit
    if missing_list == ['']:
        continue
        
    else:
        for dat_type in missing_list:  
            if azdias[column_name].dtype == 'object': #find values that contain x
                azdias.loc[azdias[column_name] == dat_type, column_name] = np.nan #replace x with nan
               
            else:
                dat_type = int(dat_type) #if no x, convert to integer and replace with nan
                azdias.loc[azdias[column_name] == dat_type, column_name] = np.nan
                
azdias.head(n=20)


# In[12]:


#find number of unknown or missing elements
code_miss_count = azdias.isnull().sum().sum()

print("The percentage of missing or unknown values is:", " ", (100*(code_miss_count/num_of_data)),"%", sep='')


# In[13]:


azdias.describe() 


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[14]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

count_miss = azdias.isnull().sum(axis=0).values #find number of nans for each column
count_miss = [val for val in count_miss]
plt.hist(count_miss, bins=20)
plt.show()


# In[15]:


# Investigate patterns in the amount of missing data in each column.

#Show stats for entire dataframe
pd.DataFrame({'stat':count_miss}).describe()


# In[16]:


#Show number of missing data in each column
pd.set_option('display.max_rows',100)
azdias.isnull().sum(axis=0) 


# In[17]:


#plot number of missing values per column

plt.figure(figsize=(20,8))
x = range(0,azdias.shape[1])
plt.bar(x,count_miss)
plt.show()


# In[18]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

#Remove columns with more than 200,000 missing values
drop_cols = []

for ind, val in enumerate(count_miss):
    if val > 200000:
        drop_cols.append(ind)
        
azdias_drop_cols = list(azdias.columns[drop_cols])
azdias = azdias.drop(azdias_drop_cols, axis=1)

azdias.head()


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# (Double click this cell and replace this text with your own text, reporting your observations regarding the amount of missing data in each column. Are there any patterns in missing values? Which columns were removed from the dataset?)
# 
# The percentage of missing data that is naturally missing is 6.46% and the percent of missing or unknown values is 11.05%. According to the histogram plot, there are a few columns which contain significantly more missing data than the other columns and so those columns were removed. Columns that were missing greater than 200,000 data points were chosen to be removed. Those columns which were removed are:
# AGER_TYP
# GEBURTSJAHR
# TITEL_KZ
# ALTER_HH
# KK_KUNDENTYP
# KBA05_BAUMAX

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[19]:


# How much data is missing in each row of the dataset?
count_miss_row = azdias.isnull().sum(axis = 1).values
plt.figure(figsize = (20,8))
ax = sns.countplot(count_miss_row, color = 'blue')
plt.show()


# In[20]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.

# Divide the data using threshold of defined variable threshold
threshold = 10 #define threshold value
azdias_sub1 = azdias.loc[count_miss_row <= threshold]
azdias_sub2 = azdias.loc[count_miss_row > threshold]

print(azdias_sub1.shape)
print(azdias_sub2.shape)


# In[21]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

count_miss_sub1_col = azdias_sub1.isnull().sum(axis = 0).values #sum number of missing values for each column

plt.figure(figsize=(20,8))
x = range(0,count_miss_sub1_col.shape[0])
plt.suptitle('Number of Missing Data Set 1')
plt.bar(x,count_miss_sub1_col)
plt.show()


# In[22]:


count_miss_sub2_col = azdias_sub2.isnull().sum(axis = 0).values #sum number of missing values for each column

plt.figure(figsize=(20,8))
x = range(0,count_miss_sub2_col.shape[0])
plt.suptitle('Number of Missing Data Set 2')
plt.bar(x,count_miss_sub2_col)
plt.show()


# In[23]:


# Use columns 5,10,15,20,25
res_col = [5,10,15,20,25]
azdias_sub1_plot = azdias_sub1.iloc[:,res_col]
azdias_sub2_plot = azdias_sub2.iloc[:,res_col]


# In[24]:


def subset_plot(ind):
    st_title = 'Column '+str(azdias_sub1_plot.columns[ind-1])+' distribution between sets 1 and 2'
    plt.suptitle(st_title)
    plt.subplot(121)
    plt.subplots_adjust(right=3, wspace=.2)
    sns.countplot(azdias_sub1_plot.iloc[:,ind - 1].values,color='blue')
    plt.subplot(122)
    sns.countplot(azdias_sub2_plot.iloc[:,ind - 1].values,color='blue')


# In[25]:


subset_plot(1)


# In[26]:


subset_plot(2)


# In[27]:


subset_plot(3)


# In[28]:


subset_plot(4)


# In[29]:


subset_plot(5)


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 
# The second subset has much more missing data per column compared to the first subset. 
# The rows seem to have much less missing data.
# There does seem to be a quantitive difference between the data with few or no missing values based on the subset plots.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[30]:


# How many features are there of each data type?

feat_info.groupby(['type']).size()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[31]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

#show only categorical type columns
feat_info[feat_info['type'] == 'categorical']


# In[32]:


# Re-encode categorical variable(s) to be kept in the analysis.

#get list of attributes with type categorical
cat_cols = [] #initialize
for i in feat_info[feat_info['type'] == 'categorical']['attribute']:
    cat_cols.append(i)


# In[33]:


#Look for one binary variable taking non-numeric and create dummy variable, and look for multi-level categoricals
#and drop if found

for cols in azdias.columns:
    if cols in cat_cols:
        if azdias[cols].nunique(dropna=True) > 2: #if the length of number of unique values is greater than 2 
            azdias = azdias.drop(cols, axis=1) #drop from the analysis
            print("more than 2 categories: {}".format(cols))
            
        else:
            if not azdias[cols].unique()[0] > 0:
                dummies = pd.get_dummies(azdias[cols], prefix=cols)
                azdias = azdias.drop(cols, axis=1) #create dummy variable
                azdias = azdias.join(dummies)
                print("transformed to dummy variable: {}".format(cols))


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# feat_info has 21 features classified as type categorical. By re-encoding the features, feat_info was decreased to 17 features due to multi-level categoricals being dropped. There was only 1 feature to be set as a dummy variable by one hot encoding, GREEN_AVANTGARDE.

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[34]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

#show only mixed type columns
feat_info[feat_info['type'] == 'mixed']


# In[35]:


azdias['PRAEGENDE_JUGENDJAHRE'].value_counts()


# In[36]:


# create variable: MOVEMENT
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([1,3,5,8,10,12,14]),'MOVEMENT'] = 1
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([2,4,6,7,9,11,13,15]),'MOVEMENT'] = 2


# In[37]:


#create variable Decade
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([1,2]), 'DECADE'] = 40
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([3,4]), 'DECADE'] = 50
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([5,6,7]), 'DECADE'] = 60
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([8,9]), 'DECADE'] = 70
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([10,11,12,13]), 'DECADE'] = 80
azdias.loc[azdias['PRAEGENDE_JUGENDJAHRE'].isin([14,15]), 'DECADE'] = 90


# In[38]:


# drop PRAEGENDE_JUGENDJAHRE column
azdias=azdias.drop('PRAEGENDE_JUGENDJAHRE',axis=1)

azdias.head()


# In[39]:


# Investigate "CAMEO_INTL_2015" and create two new variables: WEALTH and LIFE_STAGE.

feat_info[feat_info['type'] == 'mixed']

azdias['CAMEO_INTL_2015'].value_counts()


# In[40]:


azdias['CAMEO_INTL_2015']=azdias['CAMEO_INTL_2015'].astype(float)

# create new variable: WEALTH
azdias.loc[azdias['CAMEO_INTL_2015'].isin([51,52,53,54,55]), 'WEALTH'] = 1
azdias.loc[azdias['CAMEO_INTL_2015'].isin([41,42,43,44,45]), 'WEALTH'] = 2
azdias.loc[azdias['CAMEO_INTL_2015'].isin([31,32,33,34,35]), 'WEALTH'] = 3
azdias.loc[azdias['CAMEO_INTL_2015'].isin([21,22,23,24,25]), 'WEALTH'] = 4
azdias.loc[azdias['CAMEO_INTL_2015'].isin([11,12,13,14,15]), 'WEALTH'] = 5


# In[41]:


# create new variable: LIFE_STAGE
azdias.loc[azdias['CAMEO_INTL_2015'].isin([11,21,31,41,51]),'LIFE_STAGE'] = 1
azdias.loc[azdias['CAMEO_INTL_2015'].isin([12,22,32,42,52]),'LIFE_STAGE'] = 2
azdias.loc[azdias['CAMEO_INTL_2015'].isin([13,23,33,43,53]),'LIFE_STAGE'] = 3
azdias.loc[azdias['CAMEO_INTL_2015'].isin([14,24,34,44,54]),'LIFE_STAGE'] = 4
azdias.loc[azdias['CAMEO_INTL_2015'].isin([15,25,35,45,55]),'LIFE_STAGE'] = 5


# In[42]:


# drop original column
azdias = azdias.drop('CAMEO_INTL_2015',axis=1)
azdias.head()


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# For the PRAEGENDE_JUGENDJAHRE feature, two variables (MOVEMENT and DECADE) were created from the original PRAEGENDE_JUGENDJAHRE column based on Data_Dictionary.md. The PRAEGENDE_JUGENDJAHRE feature was then dropped from the dataframe.
# 
# For the CAMEO_INTL_2015 feature, two variables (WEALTH and LIFE_STAGE) were created from the original CAMEO_INTL_2015 column based on Data_Dictionary.md. The CAMEO_INTL_2015 feature was then dropped from the dataframe.
# 
# Features LP_LEBENSPHASE_FEIN and LP_LEBENSPHASE_GROB were removed as their information is already described in other variablesand so the information is redundant.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[43]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

# Remove mixed categorical variables
azdias = azdias.drop(['LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB'],axis=1)


# In[44]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

azdias.head()


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[45]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    count_miss = df.isnull().sum(axis=0).values #find number of nans for each column
    count_miss = [val for val in count_miss]
    
    drop_cols = []

    for ind, val in enumerate(count_miss):
        if val > 200000:
            drop_cols.append(ind)
        
    df_drop_cols = list(azdias.columns[drop_cols])
    df = df.drop(df_drop_cols, axis=1)
    
    for col in range(df.shape[1]):  #loop through columns
        column_name = df.columns[col]  #get column name
        missing_list = feat_info.iloc[col,3]  #get missing_or_unknown column from feature info
        missing_list = missing_list.replace('[','') #remove left bracket from string
        missing_list = missing_list.replace(']','') #remove right bracket from string
        missing_list = missing_list.split(',')  #split into individual strings
    
        #find data that is natually missing and continue loop to omit
        if missing_list == ['']:
            continue
        
        else:
            for dat_type in missing_list:  
                if df[column_name].dtype == 'object': #find values that contain x
                    df.loc[df[column_name] == dat_type, column_name] = np.nan #replace x with nan
               
                else:
                    dat_type = int(dat_type) #if no x, convert to integer and replace with nan
                    df.loc[df[column_name] == dat_type, column_name] = np.nan
                    
    # select, re-encode, and engineer column values.
    
    # encode OST_WEST_KZ
    df.loc[df['OST_WEST_KZ'] == 'W','OST_WEST_KZ'] = 0
    df.loc[df['OST_WEST_KZ'] == 'O','OST_WEST_KZ'] = 1
    
    # Re-encode categorical variable(s) to be kept in the analysis.
    
    
    #get list of attributes with type categorical
    feat_info[feat_info['type'] == 'categorical']
    
    cat_new_cols = [] #initialize
    for i in feat_info[feat_info['type'] == 'categorical']['attribute']:
        cat_new_cols.append(i)
    
    for cols in df.columns:
        if cols in cat_new_cols:
            if df[cols].nunique(dropna=True) > 2: #if the number of unique values is greater than 2 
                df = df.drop(cols, axis=1) #drop from the analysis
                print("more than 2 categories: {}".format(cols))
            
            else:
                if not df[cols].unique()[0] > 0:
                #if not df[cols].unique()[0] > 0:
                    dummies = pd.get_dummies(df[cols], prefix=cols)
                    df = df.drop(cols, axis=1) #create dummy variable
                    df = df.join(dummies)
                    print("transformed to dummy variable: {}".format(cols))
                    
    # create variable: MOVEMENT
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([1,3,5,8,10,12,14]),'MOVEMENT'] = 1
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([2,4,6,7,9,11,13,15]),'MOVEMENT'] = 2
    
    #Capture Decade
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([1,2]), 'DECADE'] = 40
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([3,4]), 'DECADE'] = 50
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([5,6,7]), 'DECADE'] = 60
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([8,9]), 'DECADE'] = 70
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([10,11,12,13]), 'DECADE'] = 80
    df.loc[df['PRAEGENDE_JUGENDJAHRE'].isin([14,15]), 'DECADE'] = 90
    
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].astype(float)

    # create new variable: WEALTH
    df.loc[df['CAMEO_INTL_2015'].isin([51,52,53,54,55]), 'WEALTH'] = 1
    df.loc[df['CAMEO_INTL_2015'].isin([41,42,43,44,45]), 'WEALTH'] = 2
    df.loc[df['CAMEO_INTL_2015'].isin([31,32,33,34,35]), 'WEALTH'] = 3
    df.loc[df['CAMEO_INTL_2015'].isin([21,22,23,24,25]), 'WEALTH'] = 4
    df.loc[df['CAMEO_INTL_2015'].isin([11,12,13,14,15]), 'WEALTH'] = 5
    
    # create new variable: LIFE_STAGE
    df.loc[df['CAMEO_INTL_2015'].isin([11,21,31,41,51]),'LIFE_STAGE'] = 1
    df.loc[df['CAMEO_INTL_2015'].isin([12,22,32,42,52]),'LIFE_STAGE'] = 2
    df.loc[df['CAMEO_INTL_2015'].isin([13,23,33,43,53]),'LIFE_STAGE'] = 3
    df.loc[df['CAMEO_INTL_2015'].isin([14,24,34,44,54]),'LIFE_STAGE'] = 4
    df.loc[df['CAMEO_INTL_2015'].isin([15,25,35,45,55]),'LIFE_STAGE'] = 5
    
    # remove selected columns and rows, ...
    df = df.drop('PRAEGENDE_JUGENDJAHRE', axis=1)
    df = df.drop('CAMEO_INTL_2015',axis=1)
    
    # Return the cleaned dataframe.
    return df


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[46]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.

imp = Imputer(missing_values=np.nan, strategy="median", axis=1) #create imputed variable
azdias_imp = pd.DataFrame(imp.fit_transform(azdias)) #apply imputer function to azdias dataframe and 
#store in new dataframe

azdias_imp.columns = azdias.columns #copy columns from azdias dataframe to imputed dataframe


# In[47]:


# Apply feature scaling to the general population demographics data.
scaler = StandardScaler()
azdias_scaled = scaler.fit_transform(azdias_imp.values) #standardize features by removing the mean and scaling to unit
#variance


# In[48]:


azdias_scaled


# ### Discussion 2.1: Apply Feature Scaling
# 
# An imputer was applied to replace all the missing values in the azdias dataframe. Then, each feature was scaled to mean 0 and standard deviation 1 by applying the StandardScaler function.
# 

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[49]:


# Apply PCA to the data.
pca = PCA()
az_pca = pca.fit(azdias_scaled)


# In[50]:


# Investigate the variance accounted for by each principal component.


# In[51]:


#create scree plot 
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(20, 8))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


# In[52]:


scree_plot(az_pca)


# In[53]:


# Re-apply PCA to the data while selecting for number of components to retain.
pca = PCA(n_components = 50)
az_pca = pca.fit_transform(azdias_scaled)


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)
# 
# Based on the scree plot function (provided in the course), choosing a principal component value of 50 yields an explained variance close to 0 so 50 components will be retained.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[54]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

#define function that takes a principal component value and creates a sorted dictionary of principal 
#component mapped to corresponding feature name
def weights(princ_comp):
    weight_dict = {}
    
    ident_mat = np.identity(azdias_scaled.shape[1])  #create identity matrix
    coef = pca.transform(ident_mat) #apply pca transformation on the identity matrix
    
    j=0
    for col in azdias.columns:
        weight_dict.update({col: coef[j][princ_comp]})  #create dictionary for each feature and associated component
        j += 1
        
    sorted_dict = sorted(weight_dict, key=weight_dict.get) #sort weighted dictionary in ascending order
    
    for item in sorted_dict:
        print(item,":", " ", weight_dict[item], sep='')


# In[55]:


#pass defined principal component to weights function
pc = 0
weights(pc)


# In[56]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

pc = 1
weights(pc)


# In[57]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

pc = 2
weights(pc)


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)
# 
# The first principal component is most positively impacted by the features PLZ8_ANTG3, PLZ8_BAUMAX, and PLZ8_ANTG4 and most negatively impacted by the features FINANZ_MINIMALIST, MOBI_REGIO, KBA05_GBZ. The PLZ features are housing related and the FINANZ, MOBI, and KBA05 features are finance related. It makes sense that they would be related positively and negatively. 
# 
# The second principal component is most positively impacted by the features ORTSGR_KLS9, MIN_GEBAEUDEJAHR, and HH_EINKOMMEN_SCORE, and most negatively impacted by the features KBA05_ANTG1, PLZ8_ANTG1, ANZ_TITEL. 
# 
# KBA05_ANTG1 and PLZ8_ANTG1 are family housing related, and AN_TITEL is the number of professional academic title holders per household. 
# 
# ORTSGR_KLS9 is the size of community, MIN_GEBAEUDEJAHR is the first year building was mentioned in database, and HH_EINKOMMEN_SCORE is the estimated household net income.
# 
# It seems that higher income earners could be living in larger, newer communities which could negatively impact number of housing.
# 
# The third principal component is most positively impacted by the features SEMIO_KRIT, ALTERSKATEGORIE_GROB, and SEMIO_ERL and most negatively impacted by the features 
# 
# SEMIO_KRIT are critically-minded individuals, ALTERSKATEGORIE_GROB is the person's estimated age, and SEMIO_ERL are event-oriented individuals SEMIO_KULT, SEMIO_TRADV, and DECADE.
# 
# SEMIO_KULT are cultural-minded individuals, SEMIO_TRADV are traditional-minded individuals, and DECADE is the feature defined above as generation defined by decade.
# 
# It seems that a person's age will have an impact as to individuals topology.

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[58]:


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)
    
    # Obtain a score related to the model fit
    score = np.abs(model.score(data))
    
    return score


# In[ ]:


# Over a number of different cluster counts...
scores = []
centers = list(range(1,30))

    # run k-means clustering on the data and...

for center in centers:
    scores.append(get_kmeans_score(az_pca, center))  # compute the average within cluster distances


# In[ ]:


# HINT: Use matplotlib's plot function to visualize this relationship.
# Investigate the change in within-cluster distance across number of clusters.

plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');


# In[ ]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.

#refit kmeans using 12 clusters
kmeans = KMeans(n_clusters = 12)
model = kmeans.fit(az_pca)

pred_pop = model.predict(az_pca)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)
# 
# I decided that based on visual observation of the SSE vs K plot that 12 clusters seems appropriate to segment the population utilizing 30 cluster counts. This was chosen as I felt that is where the elbow is located.

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[ ]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')
customers.head()


# In[ ]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

customers = clean_data(customers)
nan_customers = pd.DataFrame(customers.shape[1] - customers.count(axis=1))


# In[ ]:


#run imputer
imp_cust = Imputer(missing_values=np.nan, strategy="median", axis=1) #create imputed variable
cust_imp = pd.DataFrame(imp_cust.fit_transform(customers)) #apply imputer function to customers dataframe and 
#store in new dataframe

cust_imp.columns = customers.columns #copy columns from customers dataframe to imputed dataframe


# In[ ]:


# Apply feature scaling to the general population demographics data.
scaler = StandardScaler()
cust_scaled = scaler.fit_transform(cust_imp.values) #standardize features by removing the mean and scaling to unit
#variance

#apply PCA to the data
cust_pca = pca.fit_transform(cust_scaled)

#apply k-means clustering to the data
cust_model = kmeans.fit(cust_pca)

pred_cust = cust_model.predict(cust_pca)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[ ]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.


# In[ ]:


#Calculate the proportion of each cluster to the whole population and store in population dictionary
az_bin = np.bincount(pred_pop) #use bincount to get number of non-negative values

az_dict = {}
for p in range(len(az_bin)): #loop over number of clusters
    az_dict[p] = az_bin[p] * 100/len(azdias) #calculate percent of each cluster to the whole population


# In[ ]:


pprint.pprint(az_dict)


# In[ ]:


#Calculate the proportion of each cluster to the number of customers and store in customer dictionary
cust_bin = np.bincount(pred_cust)  #use bincount to get number of non-negative values

cust_dict={}
for q in range(len(cust_bin)):  #loop over number of clusters
    cust_dict[q] = cust_bin[q] * 100/len(customers)  #Calculate the percent of each cluster to the number of customers


# In[ ]:


pprint.pprint(cust_dict)


# In[ ]:


#plot the Demographics proportion of data points for each cluster and 
#Customers proportion of data points for each cluster.

plt.rcParams["figure.figsize"] = (20,8)
plt.subplot(1,2,1)
plt.title('General Population')
plt.ylabel('Population Percentage')
plt.xlabel('Cluster')
plt.xticks(np.arange(-1,12))
plt.yticks(np.arange(0,30,1))
plt.bar(list(az_dict.keys()),list(az_dict.values()))

plt.subplot(1,2,2)
plt.title('Customers')
plt.ylabel('Customer Percentage')
plt.xlabel('Cluster')
plt.yticks(np.arange(0,30,1))
plt.xticks(np.arange(-1,12))
plt.bar(list(cust_dict.keys()),list(cust_dict.values()))

plt.show()


# In[ ]:


#Function to calculate the centroid for defined cluster
def calc_centroid(cluster):
    centroid = scaler.inverse_transform(pca.inverse_transform(model.cluster_centers_[cluster]))
    
    center_dict = {}
    i=0

    for e in centroid:
        center_dict.update({azdias.columns[i]:e})
        i += 1
        if i == len(azdias.columns):
            break
    return center_dict


# In[ ]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?

# Calculate centroid for cluster 1 which has approximately 24.3% representation in the 
#customer data and only approximately 12.4% representation in the population data.

cent_dict_over = calc_centroid(1) #get dict of centroid for defined cluster
pprint.pprint(cent_dict_over)


# In[ ]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?

# Calculate centroid for cluster 5 which has approximately 2.8% representation in the 
#customer data and approximately 15.3% representation in the population data.

cent_dict_under = calc_centroid(5) #get dict of centroid for defined cluster
pprint.pprint(cent_dict_under)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)
# 
# My clustering analysis revealed that people in the cluster #1 customer base seem to be the most popular target audience for the company while the cluster #5 customer base seems to be the least popular target audience.
# 
# Cluster #1 represents predominately middle-aged (30-45 years old) males. They tend to have larger families with lower income. They tend to be socially minded people who tend to live in more densely populated areas with high purchasing power and low unemployment.
# 
# Cluster #5 represents predominately middle aged (30-45 years old) females. They tend to be younger, larger families with lower income. They tend to be socially minded people who tend to live in more densely populated areas. 
# 
# Based on the analysis, it seems that the company should target their male customer base as they tend to be more likely willing to shop compared to their female customer base.
# 

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




