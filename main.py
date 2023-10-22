# importing required libraries 
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from get_sentiment import main_slave
from mrk3 import *

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import adam
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_model(X_train):
    # Building the model
    model = Sequential()
    model.add(LSTM(40, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1))
    return model

def train_model(model, X_train, y_train, X_test, y_test): 
    # Model training
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=20, batch_size=1, verbose=0, shuffle=False)   
    return model

def preprocess_main(merged_final, model_path):
    merged_final['Date'] = pd.to_datetime(merged_final['Date'])
    # merged_final = merged_final.drop(["Unnamed: 0"], axis = 1)
    # creating a new dataframe to model the difference
    df_diff = merged_final.copy()

    # adding previous sales to the next row
    df_diff['prev_price'] = df_diff['Price'].shift(1)

    # dropping the null values and calculate the difference
    df_diff = df_diff.dropna()
    df_diff['diff'] = (df_diff['Price'] - df_diff['prev_price'])

    # creating new dataframe and transformating from time series to supervised
    df_supervised = df_diff.drop(['prev_price'],axis=1)

    # adding lags
    for inc in range(1,6):
        field_name = 'lag_' + str(inc)
        df_supervised[field_name] = df_supervised['diff'].shift(inc)

    # Dropping the null values
    df_supervised = df_supervised.dropna().reset_index(drop=True)

    # handling categorical data
    df_dumm = pd.get_dummies(df_supervised['Sentiment ( F/G)'])
    df_supervised = pd.concat([df_supervised, df_dumm], axis=1)
    
    df_model = df_supervised.drop(['Price','Date'],axis=1)

    df_model = df_model.drop(['Sentiment ( F/G)'], axis = 1)

    df_model.columns = ['fg', 'diff', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'Extreme Fear', 'Extreme Greed', 'Fear',	'Greed', 'Neutral']

    df_model['feargreed'] = df_model['fg']

    df_model = df_model.drop(['fg'], axis=1)

    # splitting train and test set
    train_set, test_set = df_model[0:450].values, df_model[450:].values

    # apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    # X will be all lags, y is difference column
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = get_model(X_train)

    model = train_model(model, X_train, y_train, X_test, y_test) 

    # save the model 
    model.save(model_path)  

def predict_nextday_price(model, df_model, merged_final, toconcatdf, senti_val):
    df_exp = df_model.tail(1)
    senti_list = ['Extreme Fear',	'Extreme Greed', 'Fear', 'Greed',	'Neutral']
    for senti in senti_list:
        df_exp = df_exp.drop([senti], axis = 1)

    data = [df_exp["feargreed"]]
    headers = ["feargreed"]
    df3 = pd.concat(data, axis=1, keys=headers)

    df_exp = df_exp.drop(['feargreed'], axis = 1)

    df_exp['ind'] = [0]

    toconcatdf.insert(loc = 0,
          column = 'ind',
          value = [0])

    newmerged = pd.merge(df_exp,toconcatdf,on='ind')

    newmerged = newmerged.drop(['ind'], axis = 1)

    newmerged['fg'] = df3.values[0][0]

    df_pred_data = newmerged[:].values

    # apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df_pred_data)
    # reshape training set
    future_set = df_pred_data.reshape(df_pred_data.shape[0], df_pred_data.shape[1])
    future_set_scaled = scaler.transform(future_set)

    future_set_scaled = np.delete(future_set_scaled[0], [5])

    future_set_scaled = future_set_scaled.reshape(-1,1)

    future_set_scaled = future_set_scaled.T

    future_set_scaled = future_set_scaled.reshape(future_set_scaled.shape[0], 1, future_set_scaled.shape[1])

    future_set_pred = model.predict(future_set_scaled,batch_size=1)

    future_set_pred = future_set_pred.reshape(future_set_pred.shape[0], 1, future_set_pred.shape[1])

    #rebuild whole train set for inverse transform
    pred_test_set_future = []
    pred_test_set_future.append(np.concatenate([future_set_pred[0],future_set_scaled[0]],axis=1))

    #reshape pred_test_set
    pred_test_set_future = np.array(pred_test_set_future)
    pred_test_set_future = pred_test_set_future.reshape(pred_test_set_future.shape[0], pred_test_set_future.shape[2])

    # inverse transform
    pred_test_set_future_inverted = scaler.inverse_transform(pred_test_set_future)

    difference = pred_test_set_future_inverted[0][0]

    # Next price - last price = diff
    next_day_price = merged_final.tail(1)['Price'].values[0] + difference
    print(f'Predicted Sentiment : {senti_val}')
    print(f'Next day price : {next_day_price}')
    return next_day_price, senti_val

def main(train_flag = True, train_senti_model = True):

    data_path = "csv_files\merged_final.csv"
    df_model_path = "csv_files\df_model.csv"
    model_path = "price_models\my_model"

    senti_file_path = "senti2.csv"
    senti_model_path = "sentiment_models\my_model"
    tweet = "Bitcoin prices hit all my time " 

    email = "arhampawle@gmail.com"
    pwd = "ap123"

    merged_final = pd.read_csv(data_path)
    df_model = pd.read_csv(df_model_path)

    df_model = df_model.drop(['Unnamed: 0'], axis=1)
    merged_final = merged_final.drop(["Unnamed: 0"], axis = 1)

    if train_flag:
        preprocess_main(merged_final, model_path)

    model = tf.keras.models.load_model(model_path)

    # call and take the df from other py file
    toconcatdf, senti_val = main_slave(senti_file_path, senti_model_path, tweet, train_senti_model)

    next_day_price, senti_val = predict_nextday_price(model, df_model, merged_final, toconcatdf, senti_val)

    run_gui(str(next_day_price)[:9], email, pwd, senti_val)

   

main(train_flag = True, train_senti_model = True)














