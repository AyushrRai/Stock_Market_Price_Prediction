import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load your dataset (replace 'your_dataset.csv' with your actual file)
# Assuming the dataset has columns: 'Date' and 'Close' price
data = pd.read_csv('your_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use only the 'Close' price for prediction
dataset = data[['Close']].values

# Normalize the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

# Set sequence length (how many days to look back)
seq_length = 60

# Create training and testing data
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size-seq_length:, :]

# Create sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual prices
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE metrics
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the predictions
train_plot = np.empty_like(dataset)
train_plot[:, :] = np.nan
train_plot[seq_length:len(train_predict)+seq_length, :] = train_predict

test_plot = np.empty_like(dataset)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(seq_length*2)+1:len(dataset)-1, :] = test_predict

plt.figure(figsize=(16, 8))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Price')
plt.plot(train_plot, label='Training Prediction')
plt.plot(test_plot, label='Testing Prediction')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Function to predict future prices
def predict_future(model, last_sequence, days_to_predict, scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))[0, 0]
        future_predictions.append(next_pred)
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Get the last sequence from the test data
last_sequence = X_test[-1].flatten()

# Predict next 30 days
future_days = 30
future_predictions = predict_future(model, last_sequence, future_days, scaler)

# Plot future predictions
plt.figure(figsize=(16, 8))
plt.plot(scaler.inverse_transform(scaled_data), label='Historical Price')
future_dates = pd.date_range(start=data.index[-1], periods=future_days+1)[1:]
plt.plot(future_dates, future_predictions, 'r-', label='Future Predictions')
plt.title(f'Stock Price Prediction for Next {future_days} Days')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()