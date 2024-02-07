import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU, Dense, Dropout

df = pd.read_csv('datasets_with_sentiment_Score/bitcoin_sentiments_usable_filled.csv')

label = ['Close']
price_data = df[label].values

scaler = MinMaxScaler()
price_data = scaler.fit_transform(price_data)

seq = 59

t_size = int(len(price_data) * 0.8)
data_trained = price_data[:t_size]
data_tested = price_data[t_size:]

x_train, y_train = [], []
for i in range(seq, len(data_trained)):
    x_train.append(data_trained[i-seq:i])
    y_train.append(data_trained[i][-1])
x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = [], []
for i in range(seq, len(data_tested)):
    x_test.append(data_tested[i-seq:i])
    y_test.append(data_tested[i][-1])
x_test, y_test = np.array(x_test), np.array(y_test)


# Build the hybrid model
model = Sequential()

# Add LSTM layer
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))

# Add GRU layer
model.add(GRU(50, return_sequences=True))
model.add(Dropout(0.2))

# Add another LSTM layer
model.add(LSTM(50))
model.add(Dropout(0.2))

#output layer
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test), verbose=2, shuffle=False)


#Evaluationg the model
predictions = model.predict(x_test)

test_loss = model.evaluate(x_test, y_test)

print("Test loss :", test_loss)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Mean Absolute Error (MAE) : ", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

plt.plot(y_test, label='actual')
plt.plot(predictions, label='predicted')
plt.legend()
plt.show()


#Testing the model
# Load and preprocess the dataset
df1 = pd.read_csv('59/btc_prices_59.csv')

actual_prices = df['Close'].values

# Reshape the 1D array to a 2D array
actual_prices = actual_prices.reshape(-1, 1)

total_dataset = pd.concat((df['Close'], df1['Close']),axis=0)

model_input = total_dataset[len(total_dataset) - len(df1) - seq:].values

model_input = model_input.reshape(-1,1)

model_input = scaler.fit_transform(model_input)

#Predict next day
real_data = [model_input[len(model_input)+1 - seq:len(model_input)+1,0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(prediction)