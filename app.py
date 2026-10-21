import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Underwater Wave AI",
    page_icon="🌊",
    layout="wide"
)

st.title("AI-Driven Deep Learning Simulation")
st.subheader("Underwater Wave Dynamics Prediction")

st.write(
    "Simulate underwater wave patterns and explore "
    "LSTM-based wave prediction using synthetic data."
)

# Load trained model and saved dataset
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "underwater_wave_model.keras"
    )

@st.cache_data
def load_data():
    return pd.read_csv("underwater_wave_dataset.csv")

try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Wave Simulation Controls")

amplitude = st.sidebar.slider(
    "Wave Amplitude",
    min_value=0.2,
    max_value=2.0,
    value=1.0,
    step=0.1
)

frequency = st.sidebar.slider(
    "Wave Frequency (Hz)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

noise_level = st.sidebar.slider(
    "Noise Level",
    min_value=0.0,
    max_value=0.3,
    value=0.08,
    step=0.01
)

# Generate a new synthetic wave
time = np.linspace(0, 20, 2000)

np.random.seed(42)

ideal_wave = amplitude * np.sin(
    2 * np.pi * frequency * time
)

observed_wave = ideal_wave + np.random.normal(
    0,
    noise_level,
    len(time)
)

# Display key metrics
col1, col2, col3 = st.columns(3)

col1.metric("Wave Amplitude", f"{amplitude:.2f}")
col2.metric("Wave Frequency", f"{frequency:.2f} Hz")
col3.metric("Observations", f"{len(time):,}")

# Wave simulation chart
st.header("1. Underwater Wave Simulation")

fig1, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(
    time[:500],
    ideal_wave[:500],
    label="Ideal Wave"
)

ax1.plot(
    time[:500],
    observed_wave[:500],
    label="Observed Wave",
    alpha=0.7
)

ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Wave Amplitude")
ax1.set_title("Simulated Underwater Wave")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)
plt.close(fig1)

# AI prediction using the trained model
st.header("2. LSTM Deep Learning Prediction")

st.info(
    "The trained model was trained on synthetic waves "
    "with fixed parameters. Changing the controls above "
    "changes the simulation, but does not retrain the model."
)

# Scale using the training dataset's scaler range
training_values = data["observed_wave"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(training_values)

scaled_wave = scaler.transform(
    observed_wave.reshape(-1, 1)
)

sequence_length = 30

if len(scaled_wave) > sequence_length:
    sequences = np.array([
        scaled_wave[i - sequence_length:i, 0]
        for i in range(sequence_length, len(scaled_wave))
    ])

    sequences = sequences.reshape(
        sequences.shape[0],
        sequences.shape[1],
        1
    )

    predicted_scaled = model.predict(
        sequences,
        verbose=0
    )

    predicted_wave = scaler.inverse_transform(
        predicted_scaled
    ).flatten()

    actual_wave = observed_wave[sequence_length:]

    # Evaluation on the selected synthetic wave
    mae = mean_absolute_error(
        actual_wave,
        predicted_wave
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_wave,
            predicted_wave
        )
    )

    metric1, metric2 = st.columns(2)

    metric1.metric("MAE", f"{mae:.4f}")
    metric2.metric("RMSE", f"{rmse:.4f}")

    fig2, ax2 = plt.subplots(figsize=(12, 4))

    ax2.plot(
        time[sequence_length:300 + sequence_length],
        actual_wave[:300],
        label="Actual Synthetic Wave"
    )

    ax2.plot(
        time[sequence_length:300 + sequence_length],
        predicted_wave[:300],
        label="LSTM Prediction",
        linestyle="--"
    )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Wave Amplitude")
    ax2.set_title("Actual vs Predicted Wave")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)
    plt.close(fig2)

    # Download simulation data
    output = pd.DataFrame({
        "time": time,
        "ideal_wave": ideal_wave,
        "observed_wave": observed_wave
    })

    st.download_button(
        "Download Simulation Dataset",
        data=output.to_csv(index=False),
        file_name="underwater_simulation.csv",
        mime="text/csv"
    )

st.header("3. Research Notes")

st.write(
    "This prototype uses mathematically generated wave signals "
    "and an LSTM neural network. It demonstrates a machine-learning "
    "workflow, not a validated physical model of the ocean."
)

st.success("Underwater Wave AI dashboard is ready.")
# ==========================================
# 4. TWO-DIMENSIONAL WAVE PROPAGATION
# ==========================================

st.header("4. Two-Dimensional Wave Propagation")

st.write(
    "Explore a simplified mathematical visualization "
    "of circular underwater wave propagation."
)

prop_col1, prop_col2 = st.columns(2)

with prop_col1:
    prop_amplitude = st.slider(
        "2D Wave Amplitude",
        min_value=0.2,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="prop_amplitude"
    )

with prop_col2:
    prop_frequency = st.slider(
        "2D Wave Frequency (Hz)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        key="prop_frequency"
    )

prop_time = st.slider(
    "Simulation Time (seconds)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    key="prop_time"
)

# Generate spatial grid
grid_size = 150

x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)

X, Y = np.meshgrid(x, y)

# Distance from the source
R = np.sqrt(X**2 + Y**2)
R_safe = np.maximum(R, 0.1)

# Wave parameters
wave_number = 2 * np.pi * prop_frequency
angular_frequency = 2 * np.pi * prop_frequency

# Simplified circular wave
wave_2d = (
    prop_amplitude
    * np.sin(
        wave_number * R_safe
        - angular_frequency * prop_time
    )
    / np.sqrt(R_safe)
)

# Display the 2D wave
fig3, ax3 = plt.subplots(figsize=(10, 6))

contour = ax3.contourf(
    X,
    Y,
    wave_2d,
    levels=40,
    cmap="ocean"
)

fig3.colorbar(
    contour,
    ax=ax3,
    label="Wave Amplitude"
)

ax3.set_title("2D Underwater Wave Propagation")
ax3.set_xlabel("Horizontal Distance")
ax3.set_ylabel("Vertical Distance")

st.pyplot(fig3)
plt.close(fig3)

st.caption(
    "This is an idealized mathematical simulation, "
    "not a validated physical ocean model."
)