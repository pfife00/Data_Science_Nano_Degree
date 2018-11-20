
# coding: utf-8

# Stack Overflow Survey

# <b>Questions:</b><br>
# 1) How many years of coding should I have in order to land a full-time job?<br>
# 2) How does MajorUndergrad contribute to landing a full-time job?<br>
# 3) What methods do employed developers recommend to switch to a career as a developer?<br>
# 4) Which other aspects correlate well to a developer?

# In[905]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('survey_results_public_2018.csv')  #read in data


# In[906]:


#pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', -1)
schema = pd.read_csv('survey_results_schema.csv')
schema.head(n=200)


# In[907]:


#preview the data
pd.set_option('display.max_columns', 200)
df.head(n=200)


# In[908]:


#show number of columns in dataframe
len(df.columns)


# In[909]:


df['EmploymentStatus'].unique()


# Part 1: How many years of coding do people who have a full-time coding job typically have?

# In[910]:


#Drop rows which people did not respond work full-time
df.drop(df[df.EmploymentStatus != 'Employed full-time'].index, inplace=True)


# In[911]:


#Count number of different responses
#df_yrs_prog = df_pt1_drop['YearsProgram'].value_counts(normalize=True)
df_yrs_prog = df['YearsProgram'].value_counts(normalize=True)


# In[912]:


#convert to percent
df_yrs_prog_percent = df_yrs_prog * 100


# In[913]:


#plot responses
df_yrs_prog_percent.plot.bar(figsize=(20,10), title='Percentage of Years Programming for Those Who Responded They are Employed Full-Time', rot=70);


# Part 2: Which undergraduate degree do people who have a full-time coding job typically have?

# In[914]:


#Count number of different responses using the df and the MajorUndergrad column
df_undergrad_major = df['MajorUndergrad'].value_counts(normalize=True)


# In[915]:


df_undergrad_major


# In[916]:


#convert to percent
df_undergrad_major_percent = df_undergrad_major * 100


# In[917]:


#plot responses
df_undergrad_major_percent.plot.bar(figsize=(20,10), title='Percentage of Undergradute Degree Majors for Those Who Responded They are Employed Full-Time');


# Part 3: What other aspects correlate well to those who have a full-time coding job typically have?

# In[918]:


#show schema result for the CousinEducation column since the column name is a little confusing to me
schema_cous = schema['Column'] == 'CousinEducation'
schema[schema_cous]


# In[919]:


#Count number of different responses using the df and the CousinEducation column
df_cousin = df['CousinEducation'].value_counts(normalize=True)


# In[920]:


#separate out the responses 

#Extract CousinEducation column
df_cous_col = df['CousinEducation']

#split values in each row to a list
df_split = df_cous_col.str.split(';')


# In[921]:


#Split list into separate columns and merge into new dataframe
df_cous_split = df_split.apply(pd.Series)     .merge(df_split.apply(pd.Series), left_index = True, right_index = True).dropna()


# In[922]:


#Strip values of leading white spaces
df_cous_split = df_cous_split.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# In[923]:


#Reset index
df_cous_split.reset_index(drop=True, inplace=True);


# In[924]:


#Drop the duplicate columns and rename the remaining columns
df_job_switch = df_cous_split.drop(['0_y', '1_y','2_y','3_y'], axis=1).rename(index=str, columns = {'0_x': 'Response1', '1_x': 'Response2',                                           '2_x': 'Response3',                                           '3_x': 'Response4'})


# In[925]:


df_job_switch.head()


# In[926]:


df_job_switch_counts = df_job_switch.apply(pd.Series.value_counts)


# In[927]:


#convert NaN to 0
df_job_switch_counts.fillna(0, inplace=True);


# In[928]:


#Add up values for each row and store in a Total column
df_job_switch_counts['Tot'] = df_job_switch_counts['Response1'] + df_job_switch_counts['Response2']     + df_job_switch_counts['Response3'] + df_job_switch_counts['Response4']


# In[929]:


#reset index
df_job_switch_counts.reset_index(inplace=True)


# In[930]:


#rename column
df_job_switch_counts.rename(columns={'index': 'Method'}, inplace=True)


# In[931]:


#Store the Methods and Total column to new dataframe
df_job_switch_counts_ext = df_job_switch_counts[['Method', 'Tot']]


# In[932]:


#Sort in descending order
df_job_switch_counts_ext.sort_values(by='Tot', ascending=False, inplace=True);


# In[933]:


df_job_switch_counts_ext.plot.bar(figsize=(20,10), title='Number of Recommendations by Full-Time Developers', legend=False, x='Method', rot=80);


# Part 4: What other aspects correlate well to those who have a full-time coding job typically have?

# In[934]:


#Generate correlation matrix
corr_df = df.corr()
corr_df


# In[935]:


#plot heat map
sns.heatmap(corr_df);


# In[936]:


corr_df.plot(kind='barh', figsize=(10,10), xlim=[-1,1], title='Most Correlated Traits of Full-Time Employed Developers');

