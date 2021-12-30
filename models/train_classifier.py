import sys
# import libraries for loading, preprocessing, modelling and saving
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_recall_curve,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import joblib
import pickle

import warnings
warnings.filterwarnings("ignore")



def load_data(database_filepath):
    """
    Loads clean data (cleaned_df) from the database , and defines features and label arrays for ML
    Arg:
        database_filepath : The path of the database that contain clean ETLed data
    Output: Features(X) and label (y) arrays
    
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table('cleaned_df', con=engine)
    
    #define feature(X) and targets(y)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    '''
    input: 
        raw texts and returns
    output: 
        clean_tokens- cleaned list of words in the text after normalizing, 
        tokenizing and lemmatizing
    '''
    #NORMALIZE- make lowercase, remove character not alphanumeric
    text = re.sub(r'[^A-Za-z0-9]'," ",text.lower()) 
    
    #TOKENIZE - returns a list of words in the input text
    tokens = word_tokenize(text) 
    
    #initialize lemmatizer
    lemmatizer = WordNetLemmatizer() 
    
    #stopwords removal and strip of whitespace
    stop_tokens = [word.strip() for word in tokens if word not in stopwords.words("english")]
    
    #LEMMATIZING
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in stop_tokens] 
    
    return clean_tokens


def build_model():
    """
    Arg:
       None
    Output: 
        clf_grid_model: gridSearch Model
    builds pipeline for feature extraction (techniques bag-of-words, tfidf)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('multiclf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
 
    # grid search parameters
    parameters = {'vect__binary': [False,True],
              'tfidf__norm': ['l1','l2'],
             'multiclf__estimator__estimator__C': [2.0,4.0]
             }

    # create gridsearch object and return as final model pipeline
    clf_grid_model_ppl = GridSearchCV(pipeline, param_grid=parameters)
    
    return clf_grid_model_ppl
    
    
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the evaluation report for the given model
    Input:
        model: trained model
        X_test: test data for the predication 
        Y_test: true test labels for the X_test data
    Output:
        None, but prints the evaluation report
    """ 

    #predict on test feature
    y_pred = model.predict(X_test)
    
    #print out evaluation report
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    The model we have developed is saved as a pickle file, which can be 
    used to make predicions for future data in other platforms
    
    Args:
        model: Pipeline model with a cross validated features to train
               the data
        
        model_filepath: Where the pickled model should be saved
        
    Output:
           None
    """
    
    joblib.dump(model, model_filepath)


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