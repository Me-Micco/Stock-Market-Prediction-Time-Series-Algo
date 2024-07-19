# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.graphics.tsaplots import plot_pacf
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Split into training and testing sets (Let's use an 80/20 split)

train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size, :]
test_data = data.iloc[train_size:, :]

# Fit the Prophet model on the training data
m = Prophet()
m.fit(train_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"}))

# Determine the start and end dates for the forecast
start_date = pd.to_datetime(START)
end_date = pd.to_datetime(TODAY)

# Generate the future dataframe with daily frequency
future = m.make_future_dataframe(periods=int((end_date - start_date).days), freq='D')

# Make predictions on the full future dataframe
forecast = m.predict(future)

# Extract the relevant columns from the forecast data
forecast_aligned = forecast.loc[forecast['ds'].between(start_date, end_date), ['ds', 'yhat']]

# Align the forecast data with the test data
test_data_aligned = test_data.set_index('Date')
forecast_aligned = forecast_aligned.set_index('ds')

# Align the indices and drop NaN values
common_index = test_data_aligned.index.intersection(forecast_aligned.index)
test_data_aligned = test_data_aligned.loc[common_index]
forecast_aligned = forecast_aligned.loc[common_index]
# Show and plot forecast
st.subheader('Forecast data (Prophet)')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years (Prophet)')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components (Prophet)")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(test_data_aligned['Close'], forecast_aligned['yhat'])
mse = mean_squared_error(test_data_aligned['Close'], forecast_aligned['yhat'])
rmse = np.sqrt(mse)

# Display Metrics
st.subheader('Forecast Metrics (Prophet)')
st.write("Mean Absolute Error (MAE):", mae)
st.write("Mean Squared Error (MSE):", mse)
st.write("Root Mean Squared Error (RMSE):", rmse)


# ARIMA
def check(df):
    l = []
    columns = df.columns
    for col in columns:
        dtypes = df[col].dtypes
        nunique = df[col].nunique()
        sum_null = df[col].isnull().sum()
        l.append([col, dtypes, nunique, sum_null])
    df_check = pd.DataFrame(l)
    df_check.columns = ['column', 'dtypes', 'nunique', 'sum_null']
    return df_check

df = data.copy()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

st.subheader('ARIMA')

def ad_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print('ADF:', dftest[0])
    print('P-value:', dftest[1])
    print('No. of lags:', dftest[2])
    print('Observation:', dftest[3])
    print('Critical values:')
    for key, val in dftest[4].items():
        print('\t', key, ':', val)

ad_test(df['Close'])

data_arima = df['Close']
length = int(len(df['Close']) * 0.90)
print('Length:', length)
print('Data length:', len(data_arima))
train = data_arima.iloc[:length]
print('training shape', train.shape)
test = data_arima.iloc[length:]
print('testing shape', test.shape)

plt.figure(figsize=(12, 7))
plt.title('Google Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(train, 'blue', label='Training Data')
plt.plot(test, 'green', label='Testing Data')
plt.legend()
st.pyplot(plt)

model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                             test='adf',  # use adftest to find optimal 'd'
                             max_p=3, max_q=3,  # maximum p and q
                             m=1,  # frequency of series
                             d=None,  # let model determine 'd'
                             seasonal=False,  # No Seasonality
                             start_P=0,
                             D=0,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()

import statsmodels.api as sm
pred_start = test.index[0]
pred_end = test.index[-1]
model = sm.tsa.statespace.SARIMAX(data_arima, order=(1, 1, 1))
model_fit = model.fit()
pred = model_fit.predict(start=pred_start, end=pred_end)
pred

df_sarimax = pd.DataFrame(test)
df_sarimax["prediction"] = pd.Series(pred, index=test.index)
df_sarimax.plot()
st.pyplot(plt)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate MAE
mae_arima = mean_absolute_error(test, pred) 

# Calculate MSE
mse_arima = mean_squared_error(test, pred) 

# Calculate RMSE
rmse_arima = np.sqrt(mse_arima) 
# ARIMA metrics
st.subheader('Forecast Metrics (ARIMA)')
st.write("Mean Absolute Error (MAE):", mae_arima)
st.write("Mean Squared Error (MSE):", mse_arima)
st.write("Root Mean Squared Error (RMSE):", rmse_arima)
# Define the number of days to forecast into the future
future_days = 30 # You can adjust this value as needed

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Define features (X) and target variable (y)
X = train_data[['Close']]
y = train_data['Close']
from sklearn.tree import DecisionTreeRegressor
# Implementing Decision Tree Regression Algorithm
tree = DecisionTreeRegressor().fit(X, y)

# Make predictions on the test data
tree_prediction = tree.predict(test_data[['Close']])

# Display the predictions as a table
st.subheader('Decision Tree Regressor Predictions')
st.write(pd.DataFrame({'Date': test_data['Date'], 'Predicted Close Price': tree_prediction}))

# Plot the predictions
plt.figure(figsize=(16,6))
plt.title("Decision Tree Model Prediction")
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(data['Close'], label='Original Data')
plt.plot(test_data['Date'], tree_prediction, label='Predictions', linestyle='dashed')
plt.legend()
st.pyplot(plt)

# Calculate metrics
mae_dt = mean_absolute_error(test_data['Close'], tree_prediction)
mse_dt = mean_squared_error(test_data['Close'], tree_prediction)
rmse_dt = np.sqrt(mse_dt)

# Decision Tree metrics
st.subheader('Forecast Metrics (Decision Tree Regressor)')
st.write("Mean Absolute Error (MAE):", mae_dt)
st.write("Mean Squared Error (MSE):", mse_dt)
st.write("Root Mean Squared Error (RMSE):", rmse_dt)
# LSTM


# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * .95))

# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
# Show the valid and predicted prices
st.subheader('Predicted Prices (LSTM)')

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
st.pyplot(plt)

# Show the valid and predicted prices
st.subheader('Predicted Prices (LSTM)')
st.write(valid)

# Show the RMSE value
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Calculate MAE
mae_lstm = mean_absolute_error(y_test, predictions)
# Calculate MAPE (handle potential divide by zero errors)
def mape(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / np.maximum(np.finfo(float).eps,y_true))) * 100
mape_lstm = mape(y_test, predictions)
# LSTM metrics

st.subheader('Forecast Metrics (LSTM)')

st.write("Mean Absolute Error (MAE):", mae_lstm)

st.write("Root Mean Squared Error (RMSE):", rmse)

st.write("Mean Absolute Percentage Error (MAPE):", mape_lstm) 


# Create a dictionary to hold your metrics
metrics = {
    'Prophet': {'MAE': mae, 'MSE': mse, 'RMSE': rmse},
    'ARIMA': {'MAE': mae_arima, 'MSE': mse_arima, 'RMSE': rmse_arima},
    'Decision Tree': {'MAE': mae_dt, 'MSE': mse_dt, 'RMSE': rmse_dt},
    'LSTM': {'MAE': mae_lstm, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape_lstm}   
}

# Convert the dictionary into a pandas DataFrame
metrics_df = pd.DataFrame(metrics)

# Display the DataFrame as a table in Streamlit
st.subheader('Model Comparison')
st.table(metrics_df) 



# Find the lowest values across all metrics
lowest_metrics = metrics_df.min()

# Display the best performing model and its lowest metric
best_model = lowest_metrics.idxmin()
best_metric_name = lowest_metrics.name
best_metric_value = lowest_metrics.min()

st.subheader('Best Performing Model')
st.write(f"The best model is **{best_model}** with a MAE of {best_metric_value}")