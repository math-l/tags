
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template,request,redirect,jsonify
import pandas as pd
import numpy as np
import pickle
import gzip
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn import svm


app = Flask(__name__)

# chargement du classifier multilabel
f = gzip.open('testPickleFile.pklz','rb')
classif = pickle.load(f)

# chargement du vectorizer des tags
tags_vectorizer_fitted = pickle.load( open( "/home/math31300/mysite/tags_vectorizer_fitted", "rb" ) )

# chargement vectoriser du texte
vect_fitted = pickle.load( open( "/home/math31300/mysite/word_vect_fitted", "rb" ) )

# clean tags & lemmatisation
lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

# fonctions utilisees
def clean_tags(line):
    return  BeautifulSoup(line,'lxml').get_text()

def lemmatization(line):
    return  ', '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(line)])

def lemmatize_df(line):
    line = nltk.pos_tag(w_tokenizer.tokenize(line))
    cleaned_line = []
    for word in line:
        if word[1].startswith('J'):
            pos = wordnet.ADJ
        elif word[1].startswith('V'):
            pos = wordnet.VERB
        elif word[1].startswith('N'):
            pos = wordnet.NOUN
        elif word[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = ''
        if pos != '':
            word_l = lemmatizer.lemmatize(word[0],pos=pos)
        else:
            word_l = lemmatizer.lemmatize(word[0])
        cleaned_line.append(str(word_l))
    return ', '.join(cleaned_line)


# nettoyage des contractions
contraction_clean = {
    "'m":" am",
    "'ve":" have",
    "'re":" are",
    "don't":"do not",
    "doesn't": "does not",
    "'s":" is",
    "can't": "cannot",
    "didn't": "did not",
    "isn't": "is not",
    "won't": "will not",
    "haven't": "have not",
    "shouldn't": "should not",
    "couldn't" : "could not",
    "wouldn't" : "would not",
    "aren't": "are not",
    "hasn't": "has not",
    "i'd":"i had",
    "'ll":" will",
    "wasn't":"was not"}


@app.route('/send', methods=['GET','POST'])
def send():
    if request.method == 'POST':
        title = 'Should a function have only one return statement?'
        #title = request.form['title']
        #text = request.form['text']
        text = "<p>Are there good reasons why it's a better practice to have only one return statement in a function? </p>\n\n<p>Or is it okay to return from a function as soon as it is logically correct to do so, meaning there may be many return statements in the function?</p>\n"





        d = {'Title': [title], 'Body': [text]}
        df_test = pd.DataFrame(data=d)
        #pre-traitements
        #df_test = df_test.dropna()
        #print df_test.shape
        df_test['Body'] = df_test['Body'].apply(clean_tags)

        df_test.Body = df_test['Body'].str.lower()

        df_test.Title = df_test['Title'].str.lower()

        df_test["Body"] = df_test['Body'].str.replace('[^\w\s/\']',' ')

        df_test["Title"] = df_test['Title'].str.replace('[^\w\s/\']',' ')

        df_test["Body"] = df_test['Body'].str.replace('[\\n|\_]',' ')

        df_test["Title"] = df_test['Title'].str.replace('[\\n|\_]',' ')

        df_test["Body"] = df_test['Body'].str.replace('[\s]+',' ')

        df_test["Title"] = df_test['Title'].str.replace('[\s]+',' ')

        df_test['Title'] = df_test['Title'].astype('str')

        df_test['Body'] = df_test['Body'].astype('str')

        df_test['text'] = df_test['Title']+' '+df_test['Body']

        for c in contraction_clean:
            df_test['text'] = df_test['text'].str.replace(c,contraction_clean[c])

        df_test['text'] = df_test['text'].str.replace("\'",'')
        df_test['text'] = df_test['text'].apply(lemmatize_df)
        df_test = df_test.drop(['Title', 'Body'], axis=1)

        question_test = vect_fitted.transform(df_test.text)

        question_test = pd.DataFrame(question_test.A, columns=vect_fitted.get_feature_names())
        #check = question_test.sum().sum()#type(question_test)
        final = pd.DataFrame(classif.predict(question_test.values).A, columns=tags_vectorizer_fitted.get_feature_names())

        tags = str(final.loc[0][final.loc[0] >0].keys()[0])

        #hello = df_test.Body.values[0]
        return render_template('result.html',tags=tags)

    return render_template('index.html')

