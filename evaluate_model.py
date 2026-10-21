import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 55)
print("UNDERWATER WAVE AI - MODEL EVALUATION")
print("=" * 55)

# Load the dataset
data = pd.read_csv("underwater_wave_dataset.csv")

values = data["observed_wave"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Create sequences
sequence_length = 30

X = []
y = []

for i in range(sequence_length, len(scaled_values)):
    X.append(scaled_values[i - sequence_length:i, 0])
    y.append(scaled_values[i, 0])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

# Chronological train/test split
split_index = int(len(X) * 0.8)

X_test = X[split_index:]
y_test = y[split_index:]

# Load trained model
model = tf.keras.models.load_model(
    "underwater_wave_model.keras"
)

# Predict test data
predicted_scaled = model.predict(X_test, verbose=0)

predicted = scaler.inverse_transform(predicted_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("\nMODEL PERFORMANCE")
print("-" * 35)
print(f"Test samples: {len(X_test)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Value": [mae, rmse]
})

metrics.to_csv("model_evaluation.csv", index=False)

# Plot test predictions
plt.figure(figsize=(12, 5))

plt.plot(
    actual[:300],
    label="Actual Wave"
)

plt.plot(
    predicted[:300],
    label="Predicted Wave",
    linestyle="--"
)

plt.title("LSTM Model - Test Data Evaluation")
plt.xlabel("Test Sample")
plt.ylabel("Wave Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("model_evaluation.png", dpi=150)

print("\nEvaluation saved: model_evaluation.csv")
print("Graph saved: model_evaluation.png")

plt.show()
