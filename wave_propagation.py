import numpy as np
import matplotlib.pyplot as plt

def simulate_wave_propagation(
    amplitude=1.0,
    frequency=0.5,
    duration=10,
    grid_size=150
):
    # Create a 2D spatial grid
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)

    X, Y = np.meshgrid(x, y)

    # Distance from the wave source
    R = np.sqrt(X**2 + Y**2)

    # Wave propagation parameters
    wave_number = 2 * np.pi * frequency
    angular_frequency = 2 * np.pi * frequency

    # Small value prevents division by zero
    distance = np.maximum(R, 0.1)

    # Outward-travelling circular wave
    wave = (
        amplitude
        * np.sin(wave_number * R - angular_frequency * duration)
        / np.sqrt(distance)
    )

    return X, Y, wave


def plot_wave_propagation(
    amplitude=1.0,
    frequency=0.5,
    duration=10
):
    X, Y, wave = simulate_wave_propagation(
        amplitude=amplitude,
        frequency=frequency,
        duration=duration
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(
        X,
        Y,
        wave,
        levels=40,
        cmap="ocean"
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Simulated Wave Amplitude"
    )

    ax.set_title("2D Underwater Wave Propagation")
    ax.set_xlabel("Horizontal Distance (arbitrary units)")
    ax.set_ylabel("Vertical Distance (arbitrary units)")

    return fig


if __name__ == "__main__":
    figure = plot_wave_propagation()
    figure.savefig("wave_propagation.png", dpi=150)
    plt.show()