# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridsearchCV

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
    4) Return X and Y
    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:,4]

    return X,Y

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
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    #Find best parameters
    parameters = {'clf__max_depth': [10, 20, None],
                 'clf__min_samples_leaf': [1, 2, 4],
                 'clf__min_samples_split': [2, 5, 10],
                 'clf__n_estimators': [10, 20, 40]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1)

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
    for i in range(0, len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


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
