import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 50)
print("UNDERWATER WAVE DEEP LEARNING MODEL")
print("=" * 50)

# 1. Load the simulated dataset
data = pd.read_csv("underwater_wave_dataset.csv")

values = data["observed_wave"].values.reshape(-1, 1)

# 2. Normalize the wave data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# 3. Create sequences for the LSTM
sequence_length = 30

X = []
y = []

for i in range(sequence_length, len(scaled_values)):
    X.append(scaled_values[i - sequence_length:i, 0])
    y.append(scaled_values[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. Split chronologically into training and testing data
split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length, 1)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# 6. Train the model
print("\nTraining started...")

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)

# 7. Predict the test wave values
predictions_scaled = model.predict(X_test, verbose=0)

predictions = scaler.inverse_transform(predictions_scaled)
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Evaluate model performance
mae = mean_absolute_error(actual_values, predictions)
rmse = np.sqrt(mean_squared_error(actual_values, predictions))

print("\nTRAINING COMPLETED")
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))

# 9. Save the trained model
model.save("underwater_wave_model.keras")

print("\nModel saved: underwater_wave_model.keras")

# 10. Save predictions
results = pd.DataFrame({
    "actual_wave": actual_values.flatten(),
    "predicted_wave": predictions.flatten()
})

results.to_csv("wave_predictions.csv", index=False)

print("Predictions saved: wave_predictions.csv")

# 11. Plot actual vs predicted waves
plt.figure(figsize=(12, 5))

plt.plot(
    actual_values[:300],
    label="Actual Wave",
    linewidth=2
)

plt.plot(
    predictions[:300],
    label="Predicted Wave",
    linestyle="--"
)

plt.title("LSTM Underwater Wave Prediction")
plt.xlabel("Test Sample")
plt.ylabel("Wave Amplitude (arbitrary units)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("wave_prediction.png", dpi=150)

print("Prediction graph saved: wave_prediction.png")

plt.show()