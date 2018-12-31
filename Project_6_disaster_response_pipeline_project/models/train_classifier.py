# import libraries
import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def load_data(database_filepath):
    '''
    Loads data from database created in process_data.py, stores column first columns
    in X and fourth column in Y

    Input:
    database_filepath - path to database

    Steps
    1) Create sqlite engine
    2) Utilize pandas read_sql_table function to load database
    3) Save first column to X and fourth column to Y
    4) Return X and Y and category names
    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df.iloc[:, 1]
    #Y = df.iloc[:,4]
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    
    return X,Y, category_names

def tokenize(text):
    '''
    Tokenize text and return cleaned tokenized text

    Input:
        text: Message data for tokenization

    Output:
        clean_tokens: Cleaned list after tokenization applied
    '''
    
    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build machine learning pipeline, use GridsearchCV to optimize

    Input: None

    Output:
    cv - optimized
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf', MultiOutputClassifier(RandomForestClassifier())),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
        #('clf', RandomForestClassifier()),
    ])

    #Find best parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model on test data

    Input:
    model - cleaned and trained model
    X_test - test data
    Y_test - test data
    category_names - category names
    '''
    Y_pred = model.predict(X_test)
    #Report the f1 score, precision and recall for each output category of the dataset.
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test, Y_pred))

def save_model(model, model_filepath):
    '''
    Save the model to a pickle

    Input:
    model - model
    model_filepath - filepath to model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
