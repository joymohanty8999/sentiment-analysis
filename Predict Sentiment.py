
import keras 
from keras.models import load_model
model = load_model('final_model_new.h5')



import nltk
from nltk.corpus import stopwords
import re
from keras.preprocessing.text import Tokenizer
import gensim
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import json


import os


stop_words=set(stopwords.words('english'))
for w in ['not',"couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]:
    stop_words.remove(w)



w2v_model = gensim.models.word2vec.Word2Vec(size=300, 
                                            window=7, 
                                            min_count=10, 
                                            workers=8)


with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)



def preprocess(text):
    review=re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+',' ',text)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stop_words]
    review=pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=300)
    return review

def prediction(review):
    review=preprocess(review)
    score=model.predict(review)
    score=score[0]
    if score<0.4:
        print("Angry")
    elif score>0.4 and score<0.6:
        print("Sad")
    else:
        print("Neutral")
    print(score)





prediction(" I am highly disappointed in this institutuion, the faculty are inexperienced, they do not know how to teach. The exams were conducted in a haphazard manner")





