import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils.np_utils import to_categorical
import re

def make_model(X, max_features):
  embed_dim = 500
  lstm_out = 5

#Creates a embedding layer to co relate words
  model = Sequential()
  model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
  model.add(SpatialDropout1D(0.5))
  model.add(LSTM(lstm_out, recurrent_dropout=0.4))
  model.add(Dense(5,activation='softmax'))

  return model

#Training the model 
def train(X, X_train, X_test, Y_train, Y_test, model_path, max_features):
  batch_size = 32
  model = make_model(X, max_features) 

  model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
  model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs = 20, batch_size=batch_size, verbose = 0)
  # save the model here
  model.save(model_path)

def preprocess(file_path, model_path, train_model = False):    
  senti=pd.read_csv(file_path)
  senti=senti.dropna()

  senti['text'] = senti['text'].apply(lambda x: x.lower())
  senti['text'] = senti['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

  max_features = 2000
  tokenizer = Tokenizer(num_words=max_features, split=' ')
  tokenizer.fit_on_texts(senti['text'].values)

  #padding 
  if(train_model):
    X = tokenizer.texts_to_sequences(senti['text'].values)
    X = pad_sequences(X)
    Y = pd.get_dummies(senti['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.22, random_state = 42)

    train(X, X_train, X_test, Y_train, Y_test, model_path, max_features)

  return tokenizer 


def to_dummy(sentiment_result):  
  sentiment = list(sentiment_result)
  maxlen = max(sentiment)
  
  for i in range(len(sentiment)):
    if sentiment[i] == maxlen:
      sentiment[i] = 1
      maxlen_ind = i
    else:
      sentiment[i] = 0

  senti_list_for_col = ['Extreme Fear','Extreme Greed' ,'Fear' ,'Greed' ,'Neutral']
  senti_val = senti_list_for_col[maxlen_ind]

  finaldf = pd.DataFrame(sentiment).T
  finaldf.columns = senti_list_for_col

  return finaldf, senti_val

def get_final_sentiment(tweet, tokenizer, model):   # 'Bitcoin Price Drops to $6.1K Shortly After Equities Markets Close Red'
  twt = [tweet]
  #vectorizing the tweet by the pre-fitted tokenizer instance
  twt = tokenizer.texts_to_sequences(twt)
  #padding the tweet to have exactly the same shape as embedding_2 input
  twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
  
  sentiment = model.predict(twt,batch_size=1,verbose = 1)[0]

  return to_dummy(sentiment)

def load_model(model_path):
  model = tf.keras.models.load_model(model_path)
  return model

def main_slave(file_path, model_path, tweet, train_model = True):
  # file_path = "/content/drive/MyDrive/Project/senti2.csv"
  # model_path = "/content/drive/MyDrive/Project/sentiment_models/my_model"
  # tweet = "bitcoin prices drop"

  tokenizer = preprocess(file_path, model_path, train_model)
  model = load_model(model_path)
  final_df_for_main, senti_val = get_final_sentiment(tweet, tokenizer, model)

  return final_df_for_main, senti_val

# to call this in main.py
# main_slave(train_model = False)












