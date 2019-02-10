# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from messages and categories csv files, merges dataframes

    Input:
    messages_filepath - file path for messages.csv
    categories_filepath - file path for categories.csv

    Steps
    1) Load Datasets
    2) Merge Datasets
    3) Return df
    '''

    #load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id', how='outer')

    return df

def clean_data(df):
    '''
    Takes df as input, removes duplicates and returns one dataframe (df)

    Input:
    df - loaded and merged dataframe

    Steps
    1) Split into separate category columns
    2) Convert category values to just numbers 0 or 1
    3) Replace categories column in df with new category columns
    4) Remove Duplicates
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    #category_colnames = list(map(lambda x:x[:-2], row))
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #find related categories
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(subset='id', inplace=True)

    return df

def save_data(df, database_filepath):
    '''
    Save df to sqlite database

    Input:
    df - cleaned dataframe
    database_filename - filename to save to

    Output:
    No output - saves to file
    
    Steps:
    1) Create sqlite engine
    2) Save to sql database
    '''

    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
