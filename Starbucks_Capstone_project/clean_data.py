
# coding: utf-8

# Cleaning Helper File

# This file runs applicable cleaning functions for Starbucks Capstone Project Notebook

# References
#
# https://stackoverflow.com/questions/44596077/
#     datetime-strptime-in-python

#https://scikit-learn.org/stable/modules/generated/
    #sklearn.preprocessing.MultiLabelBinarizer.html
    #sklearn.preprocessing.MultiLabelBinarizer

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import re
import progressbar


def convert_to_datetime(elem):
    '''
    Converts string to a datetime object

    INPUT:
        elem: String that stores a date in the %Y%m%d format

    OUTPUT:
        datetimeobj: Datetime object
    '''

    return datetime.strptime(str(elem), '%Y%m%d')

def update_column_name(df, old_column_name, new_column_name):
    '''
    Updates a dataframe column name

    INPUT:
        df: dataframe
        old_column_name: String that stores the old column name
        new_column_name: String that stores the new column name

    OUTPUT:
        column_names: np.array that stores the updated dataframe column names
        '''
    column_names = df.columns.values

    select_data = np.array([elem == old_column_name for elem in column_names])

    column_names[select_data] = new_column_name

    return column_names


def clean_profile_data(df):
    '''
    This function takes the profile dataframe input and returns the cleaned and
    formatted profile dataframe and another dateframe with age and gender only

    Input:
        df: profile dataframe

    Output:
        df: cleaned profile dataframe
    '''

    #Drop rows containing nulls based on specified column
    df.dropna(axis=0, inplace=True)

    #If statement to clean profile dataframe
    if 'age' in df.columns:
        #drop age values of 118
        df.drop(df[df.age == 118].index, inplace = True)

    #create new dataframe for gender column to perform one hot encoding
    profile_gender = df['gender']

    #one hot encode gender and add back to original dataframe
    df = pd.concat([df, pd.get_dummies(profile_gender)], axis=1)

    #reset index
    df = df.reset_index(drop=True)

    #rename id column to customer_id for consistency with transcript DataFrame
    df.rename(columns={'id': 'customer_id'}, inplace=True)

    # Transform the became_member_on column to a datetime object
    df['became_member_on'] = df['became_member_on'].apply(convert_to_datetime)

    #extract year from the became_member_on column and store in new
    #membership_start_year column
    df['membership_start_year'] = df['became_member_on'].apply(lambda elem: elem.year)

    #one hot encode membership start year
    membership_start_year_df = pd.get_dummies(df['membership_start_year'])

    # One hot encode a customer's age by categorizing in a range
    min_age_limit = np.int(np.floor(np.min(df['age'])/10)*10)
    max_age_limit = np.int(np.ceil(np.max(df['age'])/10)*10)

    df['age_range'] = pd.cut(df['age'],
               (range(min_age_limit,max_age_limit + 10, 10)), right=False)

    #convert age range column  to string
    df['age_range'] = df['age_range'].astype('str')

    #one hot encode
    age_range_df = pd.get_dummies(df['age_range'])

    #set ordering for customer_id and income columns to move to front
    column_ordering = ['customer_id', 'income']

    #move columns to front
    column_ordering.extend(age_range_df.columns.values)
    column_ordering.extend(membership_start_year_df.columns.values)

    # appened one hot encoded age range year variables
    df = pd.concat([df, age_range_df], axis=1)

    #copy age, gender, and income columns to another dataframe
    age_gender_income_df = df[['age', 'M', 'F', 'income']].copy()

    #drop uncessary columns due to one hot encoding
    df.drop(['gender', 'O', 'age', 'age_range', 'became_member_on'], axis=1, inplace=True)

    return df, age_gender_income_df


def clean_portfolio_data(df):
    '''
    Transforms dataframe to make column meanings more clear, one hot encodes offer types

    Input:
        df: portfolio dataframe

    Output:
        df: cleaned dataframe containing extracted offer ids
    '''

    #change name of id column to offer_id, duration to days_duration
    df.rename(columns={'id': 'offer_id', 'duration': 'days_duration'}, inplace=True)

    #create new dataframe for offer_type column to perform one hot encoding
    offer_df = df['offer_type']

    #one hot encode offer type and add back to original dataframe
    df = pd.concat([df, pd.get_dummies(offer_df)], axis=1)

    #apply MultiLabelBinarizer function to one hot encode channels column
    mlb = MultiLabelBinarizer()
    mlb.fit(df['channels'])
    channel_df = pd.DataFrame(mlb.transform(df['channels']), columns=mlb.classes_)

    #add to original dataframe
    #df = pd.concat([df, channel_df], sort=False)

    # Replace the 'offer_type' and 'channels' columns
    df = pd.concat([df, offer_df, channel_df], axis=1)

    #drop offer_type column from original dataframe
    df.drop(['offer_type', 'channels'], axis=1, inplace=True)

    #redorder columns
    df = df[['offer_id', 'bogo', 'discount', 'informational', 'difficulty', 'days_duration', 'reward', 'email',
                 'mobile', 'social', 'web']]

    return df

def clean_transcript_data(df, profile):
    '''
    Transforms transcript and profile dataframes to make column meanings more clear, remove customer ids
    that don't match with profile dataframe, splits into two dataframes: one for offers, one for transactions only

    Input:
        df, profile: transaction data and profile dataframes

    Output:
        offer_df, df: Cleaned offer_df which is dataframe of offers transactions and transactions which is df of just
        purchases with no offers
    '''

    # Change the name of the 'person' column to 'customer_id'
    df.columns = update_column_name(df, 'person', 'customer_id')

    # Remove customer id's that are not in the customer profile DataFrame
    select_data = df['customer_id'].isin(profile['customer_id'])
    df = df[select_data]

    percent_removed = 100 * (1 - select_data.sum() / select_data.shape[0])
    print('Percentage of transactions removed: %.2f %%' % percent_removed)

    # Convert from hours to days
    df['time'] /= 24.0

    # Change the name of the 'time' column to 'timedays'
    df.columns = update_column_name(df, 'time','days')

    # Select customer offers
    pattern_obj = re.compile('^offer (?:received|viewed|completed)')
    h_is_offer = lambda elem: pattern_obj.match(elem) != None
    is_offer = df['event'].apply(h_is_offer)

    offer_data = df[is_offer].copy()
    offer_data = offer_data.reset_index(drop=True)

    # Initialize a list that describes the desired output DataFrame column ordering
    column_order = ['offer_id', 'customer_id', 'days']

    # Create an offerid column
    offer_data['offer_id'] =\
        offer_data['value'].apply(lambda elem: list(elem.values())[0])

    # Transform a column that describes a customer offer event
    pattern_obj = re.compile('^offer ([a-z]+$)')

    h_transform = lambda elem: pattern_obj.match(elem).groups(1)[0]

    offer_data['event'] = offer_data['event'].apply(h_transform)


    # One hot encode customer offer events
    event_df = pd.get_dummies(offer_data['event'])
    column_order.extend(event_df.columns.values)

    # Create a DataFrame that describes customer offer events
    offer_data = pd.concat([offer_data, event_df], axis=1)
    offer_data.drop(columns=['event', 'value'])
    offer_data = offer_data[column_order]

    # Select customer transaction events
    df = df[is_offer == False]
    df = df.reset_index(drop=True)

    # Transform customer transaction event values
    df['amount'] = df['value'].apply(lambda elem: list(elem.values())[0])

    # Create a DataFrame that describes customer transactions
    df = df.drop(columns=['event', 'value'])
    column_order = ['customer_id', 'days', 'amount']
    df = df[column_order]

    return offer_data, df


def create_combined_records(customer_id, portfolio, profile, offer_df, transaction_df):
    '''
    Creates a list of dictionaries that describes the effectiveness of
    offers to a specific customer

    Input:
        customer_id: String that refers to a specific customer
        profile: profile dataframe
        portfolio: portfolio dataframe
        offer_df: offer_df dataframe created from clean_transcript_data function
        transaction_df: transaction_df created from clean_transcript_data function

    OUTPUT:
        rows: list of dictionaries that describes the effectiveness of
        offers to a specific customer
    '''
    # Select a customer's profile
    current_customer = profile[profile['customer_id'] == customer_id]

    # Select offer data for a specific customer from the offer_df dataframe
    select_offer_data = offer_df['customer_id'] == customer_id
    customer_offer_data = offer_df[select_offer_data]

    #drop customer_id column and reset index
    customer_offer_data = customer_offer_data.drop(columns='customer_id')
    customer_offer_data = customer_offer_data.reset_index(drop=True)

    # Select transactions for same specific customer from transaction_df dataframe
    select_transaction = transaction_df['customer_id'] == customer_id
    customer_transaction_data = transaction_df[select_transaction]

    #drop customer_id column and reset index
    customer_transaction_data = customer_transaction_data.drop(columns='customer_id')
    customer_transaction_data = customer_transaction_data.reset_index(drop=True)

    # Initialize DataFrames that describe when a customer receives, views, and completes an offer
    event_type = ['completed', 'received', 'viewed']

    #match where the offer received data equals 1 from offer_df dataframe, drop duplicate columns and reset index
    offer_received = customer_offer_data[customer_offer_data['received'] == 1]
    offer_received = offer_received.drop(columns=event_type)
    offer_received = offer_received.reset_index(drop=True)

    #match where the offer viewed data equals 1 from offer_df dataframe, drop duplicate columns and reset index
    offer_viewed = customer_offer_data[customer_offer_data['viewed'] == 1]
    offer_viewed = offer_viewed.drop(columns=event_type)
    offer_viewed = offer_viewed.reset_index(drop=True)

    #match where the offer completed data equals 1 from offer_df dataframe, drop duplicate columns and reset index
    offer_completed = customer_offer_data[customer_offer_data['completed'] == 1]
    offer_completed = offer_completed.drop(columns=event_type)
    offer_completed = offer_completed.reset_index(drop=True)

    # Iterate over each offer a customer receives
    rows = []
    for idx in range(offer_received.shape[0]):

        # Initialize the current offer id
        current_offer_id = offer_received.iloc[idx]['offer_id']

        # Look-up a description of the current offer in portfolio df
        current_offer = portfolio.loc[portfolio['offer_id'] == current_offer_id]
        time_offer = current_offer['days_duration'].values[0]

         # Initialize the time period when an offer is valid from offer_df dataframe
        current_offer_start_time = offer_received.iloc[idx]['days']
        current_offer_end_time = offer_received.iloc[idx]['days'] + time_offer

        # Initialize a boolean array that select customer transcations that
        # fall within the valid offer time window from transaction_df and offer_df dataframes
        select_transaction = np.logical_and(customer_transaction_data['days'] >=
                           current_offer_start_time, customer_transaction_data['days'] <=
                           current_offer_end_time)

        # Initialize a boolean array that selects a description of when a
        # customer completes an offer
        select_offer_completed = np.logical_and(offer_completed['days'] >= current_offer_start_time,
                           offer_completed['days'] <= current_offer_end_time)

         # Initialize a boolean array that selects a description of when a
        # customer views an offer
        select_offer_viewed = np.logical_and(offer_viewed['days'] >= current_offer_start_time,
                           offer_viewed['days'] <= current_offer_end_time)

        # Determine whether the current offer was successful
        current_offer_successful = select_offer_completed.sum() > 0 and select_offer_viewed.sum() > 0

        # Select customer transcations that occurred within the current offer valid time window
        current_offer_transactions = customer_transaction_data[select_transaction]

        # Initialize a dictionary that describes the current customer offer
        current_row = {'offer_id': current_offer_id, 'customer_id': customer_id,
                       'days': current_offer_start_time,
                       'completed': int(current_offer_successful),
                       'total_amount': current_offer_transactions['amount'].sum()}

        current_row.update(current_offer.iloc[0,1:].to_dict())

        current_row.update(current_customer.iloc[0,1:].to_dict())

        # Update a list of dictionaries that describes the effectiveness of offers to a specific customer
        rows.append(current_row)

    return rows


def create_offer_analysis_dataset(profile, portfolio, offer_df, transaction_df):

    '''
    Combines profile, portfolio, offer_df, and transaction_df dataframes into one
    dataframe

    INPUT:
        profile: profile dataframe
        portfolio: portfolio dataframe
        offer_df: offer_df dataframe created from clean_transcript_data function
        transaction_df: transaction_df created from clean_transcript_data function

    OUTPUT:
        clean_data: combined and cleaned dataframe
        '''

    clean_data = []

    #create dataframe of unique customer ids
    customerid_list = offer_df['customer_id'].unique()

    for idx in range(len(customerid_list)):

        clean_data.extend(create_combined_records(customerid_list[idx], portfolio, profile,
                                                  offer_df, transaction_df))

    #convert to dataframe
    clean_data = pd.DataFrame(clean_data)

    #drop email, mobile and web columns as they contain all NaN values
    clean_data.drop(columns=['email', 'mobile', 'web', 'social', 'customer_id', 'days'], inplace=True)

    return clean_data.reset_index(drop=True)
