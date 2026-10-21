import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# UNDERWATER WAVE DYNAMICS SIMULATION
# ==========================================

print("=" * 50)
print("UNDERWATER WAVE DYNAMICS SIMULATION")
print("=" * 50)

# Simulation settings
sampling_rate = 100
duration = 20

# Time values
time = np.linspace(
    0,
    duration,
    sampling_rate * duration
)

# Wave parameters
amplitude = 1.0
frequency = 0.5
phase = 0.0

# Simulated underwater wave signal
wave = amplitude * np.sin(
    2 * np.pi * frequency * time + phase
)

# Add small random disturbances
np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=0.08,
    size=len(time)
)

observed_wave = wave + noise

# Save the simulation dataset
dataset = pd.DataFrame({
    "time": time,
    "ideal_wave": wave,
    "observed_wave": observed_wave
})

dataset.to_csv(
    "underwater_wave_dataset.csv",
    index=False
)

# Display basic information
print("\nSimulation completed successfully!")
print("Total observations:", len(dataset))
print("Dataset saved: underwater_wave_dataset.csv")

print("\nFirst five observations:")
print(dataset.head())

# Plot the simulated waves
plt.figure(figsize=(12, 5))

plt.plot(
    time,
    wave,
    label="Ideal Wave",
    linewidth=2
)

plt.plot(
    time,
    observed_wave,
    label="Observed Wave with Noise",
    alpha=0.7
)

plt.title("Underwater Wave Dynamics Simulation")
plt.xlabel("Time (seconds)")
plt.ylabel("Wave Amplitude (arbitrary units)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("wave_simulation.png", dpi=150)

print("\nGraph saved: wave_simulation.png")

plt.show()
