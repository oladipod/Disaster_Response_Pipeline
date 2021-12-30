#ETL process
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import sqlite3
from sqlalchemy import create_engine

#EXTRACTION
def load_data(messages_filepath, categories_filepath):
    """
    loads both messages and categories dataset, then merges them 
    into a single df.
    Args:
        messages_filepath: Absolute filepath to the messages dataset
        categories_filepath: Absolute filepath to the categories dataset
    Output: 
        df: the merged df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on='id')
    return df
    
    
#TRANSFORMATION
def clean_data(df):
    """
    Data transformation process. Split categories col into separate
    columns, renames the columns and then edits the column values 
    into digits only, accordingly
    Args:
        df: the raw merged df
    Output: 
        df: the cleaned and preprocessed df
    """
    # split categories column into separate other columns
    categories =  df['categories'].str.split(';', expand=True)
    
    #use first entry of df dataframe to generate titles for the newly generated cols
    category_colnames = df['categories'].iloc[0].replace("-0", '')\
    .replace("-1", '').split(';')
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x.split("-")[-1]).astype('int')
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True) 
    
    # replace digit 2 with digit 1
    categories['related']=categories['related'].replace(2,1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df= pd.concat([df, categories], axis = 1)
    
    # Removing duplicates
    df= df[~df.duplicated()]
    
    return df
    
    
#LOADING
def save_data(df, database_filename):
    """
    Saves the cleaned/transformed dataset into an sqlite database. The cleaned 
    data will later serve as input to the NLP/ML pipeline
    Args: 
        df: the cleaned transformed dataframe
        database_filename: database to store the cleaned dataframe 
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('cleaned_df', engine, index=False, if_exists='replace')

    
    
#
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