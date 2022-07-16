#!pip install yfinance --upgrade --no-cache-dir

import json
import pandas as pd
import numpy as np
import types
import streamlit as st
from datetime import date
import fix_yahoo_finance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import tensorflow as tf
from tensorflow import keras
from PIL import Image

START = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

image = Image.open("ForPredx.png")
st.image(image)
st.title("ForPredx App")

stocks = ("EURUSD=X", "GBPUSD=X","SGDUSD=X")
selected_stocks= st.selectbox("Select currency pair to predict", stocks)

#n_years = st.slider("Years of prediction:", 1, 4)
#period = n_years*365

#@st.cache
def load_data(ticker):
    data= yf.download(ticker,START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state= st.text("Loading in progress...")
data = load_data(selected_stocks)
data_load_state.text("Loading complete!")

st.subheader('Latest market data for the last 5 days')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='price_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='price_close'))
    fig.layout.update(title_text="Current year trend since 2021", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

plot_raw_data()

#Forecast with ForPredx
#df_EURUSD = pd.read_csv("220524-220607_EURUSD_historical_data.csv")
#df_EURUSD = df_EURUSD[df_EURUSD.CLOSE != "."]

from datetime import date
import math

d0 = date(2021, 1, 1)
d1 = date.today()

delta = d1 - d0

df_EURUSD=data[data.Close !="."] 
df_EURUSD= np.reshape(df_EURUSD,((round(delta.days/1.398)), 7))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import load_model

batch_size = 10
epochs = 300
timesteps = 5

def get_train_length(dataset, batch_size, test_percent):
    # substract test_percent to be excluded from training, reserved for testset
    length = len(dataset)
    length *= 1 - test_percent
    train_length_values = []
    for x in range(int(length) - 100,int(length)): 
        modulo=x%batch_size
        if (modulo == 0):
            train_length_values.append(x)
            print(x)
    return (max(train_length_values))

length = get_train_length(df_EURUSD, batch_size, 0.1)
#print(length)

#Adding timesteps * 2
upper_train = length + timesteps*2
df_EURUSD_train = df_EURUSD[0:upper_train]
training_set = df_EURUSD_train
training_set = df_EURUSD_train.iloc[:,1:2].values
#training_set.shape

# Feature Scaling
#scale between 0 and 1. the weights are easier to find.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(np.float64(training_set))
#training_set_scaled.shape

X_train = []
y_train = []

# Creating a data structure with n timesteps

#print(length + timesteps)
for i in range(timesteps, length + timesteps): 
    X_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i:i+timesteps,0])

#print(len(X_train))
#print(len(y_train))

#create X_train matrix
#5 items per array (timestep) 
print(X_train[0:2])
print(np.array(X_train).shape)
#create Y_train matrix
#5 items per array (timestep) 
#print(y_train[0:2])
#print(np.array(y_train).shape)

# Reshaping
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
#print(X_train.shape)
#print(y_train.shape)

# Building the LSTM
# Importing the Keras libraries and packages

from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
import h5py

# Initialising the LSTM Model with MSE Loss Function

inputs_1_mse = Input(batch_shape=(batch_size,timesteps,1))
lstm_1_mse = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mse)
lstm_2_mse = LSTM(10, stateful=True, return_sequences=True)(lstm_1_mse)

output_1_mse = Dense(units = 1)(lstm_2_mse)

regressor_mse = Model(inputs=inputs_1_mse, outputs = output_1_mse)

#mse -> mean squared error as loss function
regressor_mse.compile(optimizer='adam', loss = 'mse')
regressor_mse.summary()

import tensorflow as tf
from keras import backend as K

K.set_session(K.tf.compat.v1.Session(config=K.tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

#Stateful
epochs = 300
for i in range(epochs):
    print("Epoch: " + str(i))
    regressor_mse.fit(X_train, y_train, shuffle=False, epochs = 1, batch_size = batch_size)
    regressor_mse.reset_states()
    
#Stateless
#between the batches the cell and hidden states are lost.
#regressor_mae.fit(X_train, y_train, shuffle=False, epochs = epochs, batch_size = batch_size)

#save model
import h5py
regressor_mse.save(filepath="EURUSD_with_mse_30_ts.h5")

#load model
import h5py
regressor_mse = load_model(filepath="EURUSD_with_mse_30_ts.h5")

def get_test_length(dataset, batch_size):
    
    test_length_values = []
    for x in range(len(dataset) - 200, len(dataset) - timesteps*2): 
        modulo=(x-upper_train)%batch_size
        if (modulo == 0):
            test_length_values.append(x)
            print(x)
    return (max(test_length_values))

test_length = get_test_length(df_EURUSD, batch_size)
print(test_length)
upper_test = test_length + timesteps*2
testset_length = test_length - upper_train
#print(testset_length)

#print(upper_train, upper_test, len(df_EURUSD))

# construct test set

#subsetting
df_EURUSD_test = df_EURUSD[upper_train:upper_test] 
test_set = df_EURUSD_test.iloc[:,1:2].values

#scaling
scaled_real_bcg_values_test = sc.fit_transform(np.float64(test_set))

#creating input data
X_test = []
for i in range(timesteps, testset_length + timesteps):
    X_test.append(scaled_real_bcg_values_test[i-timesteps:i, 0])
X_test = np.array(X_test)

#reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#print(X_test)
#X_test.shape

#prediction
predicted_bcg_values_test_mse = regressor_mse.predict(X_test, batch_size=batch_size)
regressor_mse.reset_states()

predicted_bcg_values_test_mse = np.reshape(predicted_bcg_values_test_mse, 
                                       (predicted_bcg_values_test_mse.shape[0], 
                                        predicted_bcg_values_test_mse.shape[1]))
predicted_bcg_values_test_mse = sc.inverse_transform(predicted_bcg_values_test_mse)

pred_mse = []

for j in range(0, testset_length - timesteps):
    pred_mse = np.append(pred_mse, predicted_bcg_values_test_mse[j, timesteps-1])

pred_mse = np.reshape(pred_mse, (pred_mse.shape[0], 1))

# Visualising the results
plt.plot(test_set[timesteps:len(pred_mse)].astype(float), color = 'red', label = 'Real EUR/USD Prices')
plt.plot(pred_mse[0:len(pred_mse) - timesteps], color = 'green', label = 'Predicted EUR/USD with MSE')
plt.title('EUR/USD Prediction - MSE')
plt.xlabel('Time')
plt.ylabel('EUR/USD')
plt.legend()
plt.show()

#The plot
import plotly.express as px

st.subheader("Predicted FX Prices using ForPredx")
def plot_predicted_data():
    df1 = pd.DataFrame(pred_mse)
    df1 = df1.tail(5)
    fig4 = px.line(df1, x=df1.index, y=0, title='Predicted FX Prices for the next 5 days')
    fig4.layout.update(title_text="Predicted FX Prices for the next 5 days", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig4)

plot_predicted_data()

df1 = pd.DataFrame(pred_mse)
df1=df1.tail(5)
df1
