import json
import plotly
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, jsonify
from plotly.graph_objs import Bar, Layout, Figure
import joblib
from sqlalchemy import create_engine
from flask import request




app = Flask(__name__)

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



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    """
    CREATE VISUALISATIONS USING PLOTLY
    """
    #First, extract the data variables that will be useful to plot the graphs
    #1.Variables to show the distribution of genre column
    genre_counts = df.groupby('genre').count()['id']
    genre_names = list(genre_counts.index)

    #2.Variables to show distribution of different category columns
    categories_list = list(df.columns[4:])
    categories_counts = [np.sum(df[category]) for category in categories_list]

    # extract features df
    categories_df = df.iloc[:,4:]
    categories_mean = categories_df.mean().sort_values(ascending=False)[1:8] #top 7 avg 
    categories_names = list(categories_mean.index)
    
    
    # For a start, let us create 3 different graphs and save them in the 'graphs' list variable
    graphs = []
    
    #Graph 1- Show distribution of the different message categories
    #this list will eventually be the value to key 'data'
    graph_one = list(Bar(
        x = categories_list,
        y = categories_counts
        ))
    
    #this will be passed as dictionary to the 'layout' key.
    layout_one = Layout(title = 'Distribution of Message Categories',
                        xaxis ={'title' : 'Message Category'},
                        yaxis ={'title': 'Message Count'}
                       )
    
    graphs.append({'data':graph_one, 'layout':layout_one})

    
    #Graph 2- Show the top 7 message categories 
    graph_two = list(Bar(
        x = categories_names,
        y = categories_mean
    ))
    
    #this will be passed as dictionary to the 'layout' key.
    layout_two = Layout(title = 'Top 7 message Categories (Mean-Based)',
                        xaxis ={'title' : 'Message Category'},
                        yaxis ={'title': 'Avg no of messages'}
                       )
    
    graphs.append({'data':graph_two, 'layout':layout_two})

    
    #Graph 3- Show distribution of the genre categories 
    graph_three = list(Bar(
        x = genre_names,
        y = genre_counts
    ))
    
    #this will be passed as dictionary to the 'layout' key.
    layout_three= Layout(title = 'Distribution of Message Genre',
                        xaxis ={'title' : 'Count'},
                        yaxis ={'title': 'Genre'}
                       )
    
    graphs.append({'data':graph_three, 'layout':layout_three})
    #end of graphs
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()