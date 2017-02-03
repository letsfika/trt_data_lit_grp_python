# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:13:08 2017

@author: marga
"""


import pandas as pd
pd.set_option('display.max_colwidth', -1)
tweet_df = pd.read_csv("C:/Users/marga/Desktop/tweet_sample.csv", delimiter=",", encoding = "utf-8")

# A look at the dimension of the dataframe
tweet_df.shape

# A look at the columns of the dataframe
tweet_df.columns.values

# A look at sample data
tweet_df[0:5]

# We can also look at a ramdom sample of the rows
tweet_df.sample(5)

# Let's use one tweet as an example
tweet_eg = tweet_df['tweet_body'][8]
tweet_eg

# Regular Expression itself is a very useful skill to learn.
import re

# A lot of the tweets contains reference urls, we want to remove them first
def remove_url(text):
	text = re.sub('http://[^ ]*', '', text)
	text = re.sub('https://[^ ]*', '', text)
	return text

# Using the function on our sample tweet
tweet_eg = remove_url(tweet_eg)
tweet_eg 

# Removing the at users
def remove_at_user(text):
	return re.sub('@[^\s]+','', text)
 
tweet_eg = remove_at_user(tweet_eg)
tweet_eg

# Now try to write a function to remove the retweet 'RT'
def remove_rt(text):
    text = re.sub('RT', '', text, count=1)
    return text
    
tweet_eg = remove_rt(tweet_eg)
tweet_eg

# Let's remove the punctuations and numbers, basically all the non letters for now
def remove_non_letters(text):
	return re.sub('[^a-zA-Z]', ' ', text) 	
 
tweet_eg = remove_non_letters(tweet_eg)
tweet_eg

# We might want to remove some extra blanks
def remove_extra_blanks(text):
	text = re.sub('\n', ' ', text)
	text = re.sub(" +"," ",text).strip() #remove extra spaces
	return text
 
tweet_eg = remove_extra_blanks(tweet_eg)
tweet_eg

# Standardizing Cases
def all_lower_case(text):
	return text.lower()

tweet_eg = all_lower_case(tweet_eg)
tweet_eg

# Now, let's put all of the above cleaning functions together
def my_text_cleanser(text):
    if isinstance(text,str):
        #text = text.encode('utf-8')
        text = remove_url(text)
        text = remove_rt(text)
        text = remove_non_letters(text)
        text = remove_extra_blanks(text)
        text = all_lower_case(text)
        return text

# We will apply the text cleanser to our 'tweet_body' column, using a very commonly used function in pandas 'apply'
tweet_df['tweet_body_clean'] = tweet_df.tweet_body.apply(my_text_cleanser)

# Take a look at the old column and the cleaned new column
tweet_df[['tweet_body','tweet_body_clean']].sample(5)

# If you haven't done so already, download the nltk's corpus for stopwords
import nltk
nltk.download()

# Import the stop word list
from nltk.corpus import stopwords 
print (stopwords.words("english")) 

def remove_stopwords(text):
    words = text.split()
    meaningful_words = [w for w in words if not w in stopwords.words("english") ]
    return meaningful_words
    
tweet_eg = remove_stopwords(tweet_eg)
tweet_eg

# Examples of stemmed words
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
print (snowball_stemmer.stem('interaction'))
print (snowball_stemmer.stem('interact'))
print (snowball_stemmer.stem('interactions'))
print (snowball_stemmer.stem('interactivity'))

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
print (wordnet_lemmatizer.lemmatize('interaction'))
print (wordnet_lemmatizer.lemmatize('interact'))
print (wordnet_lemmatizer.lemmatize('interactions'))
print (wordnet_lemmatizer.lemmatize('interactivity'))

# We will be using the lemmatizer for our purpose
def lemmatizer(words):
    return [wordnet_lemmatizer.lemmatize(w) for w in words]
    
def my_text_tokenizer(text):
    words = remove_stopwords(text)
    words = lemmatizer(words)
    return words

# Now let's apply the functions above to our cleaned tweet
tweet_df['tweet_body_terms'] = tweet_df.tweet_body_clean.apply(my_text_tokenizer)

# Take a look at what we've done so far
tweet_df[['tweet_body','tweet_body_clean','tweet_body_terms']].sample(5)

# Since we will be creating statistics at user level, we group the dataframe by users
users_df = tweet_df.groupby('handle').agg({'tweet_body_terms':sum,'tweet_body_clean':lambda x: ' '.join(x)})

# Find top collocation in the tweets
from nltk.collocations import BigramCollocationFinder

def top_collocation_text(words):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(5)
    return finder.nbest(bigram_measures.pmi, 20)
    
# Let's see what are the most often talked about terms for Hilary and Trump
users_df['top_collocation_text'] = users_df.tweet_body_terms.apply(top_collocation_text)

print (users_df['top_collocation_text'])

def lexical_diversity(words):
    return 1.0*len(set(words))/(len(words)+1)
    
users_df['lexical_diversity'] = users_df.tweet_body_terms.apply(lexical_diversity)

print (users_df['lexical_diversity'])