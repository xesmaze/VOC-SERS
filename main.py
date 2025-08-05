import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def load_sample(file_path, center=True):
    import pandas as pd
    import numpy as np

    # Load file and skip commented lines
    df = pd.read_csv(file_path, sep="\t", comment="#", header=None)

    # Drop the first row (non-numeric headers or metadata)
    df = df.iloc[1:].reset_index(drop=True)

    # Extract wavenumbers and intensities
    wavenumbers = df.iloc[:, [0, 2, 4]]
    intensities = df.iloc[:, [1, 3, 5]]

    # Convert to numeric
    wavenumber_vals = wavenumbers.iloc[:, 0].astype(float).values
    intensity_vals = intensities.apply(pd.to_numeric, errors='coerce')

    # Optional: mean-center across replicates (columns)
    if center:
        intensity_vals = intensity_vals - intensity_vals.mean(axis=0)

    # Convert to NumPy array
    intensity_vals = intensity_vals.to_numpy()

    return wavenumber_vals, intensity_vals

def fit_gpr(wavenumbers, intensities, label, color):
    # Stack replicates for GPR
    x = np.tile(wavenumbers, intensities.shape[1]).reshape(-1, 1)
    y = intensities.T.flatten()

    # Fit GPR
    kernel = RBF(length_scale=100.0, length_scale_bounds=(1e-6, 1e3)) + WhiteKernel(noise_level=0.01)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(x, y)

    # Predict smooth curve
    x_pred = np.linspace(wavenumbers.min(), wavenumbers.max(), 1000).reshape(-1, 1)
    y_pred, y_std = gpr.predict(x_pred, return_std=True)

    # Plot raw replicate measurements
    for i in range(intensities.shape[1]):
        plt.plot(wavenumbers, intensities[:, i], alpha=0.3, label=f"{label} – Rep {i+1}", color='gray')

    # Plot GPR mean and confidence interval
    plt.plot(x_pred.ravel(), y_pred, color=color, label=f"GPR mean – {label}")
    plt.fill_between(x_pred.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                     color=color, alpha=0.2, label=f"95% CI – {label}")

    return x_pred, y_pred, y_std

# --- Sample 0 ---
wavenumbers_0, intensities_0 = load_sample("Sample 0.txt")

# --- Sample 3B ---
wavenumbers_3b, intensities_3b = load_sample("Sample 3B.txt")

# --- Sample 3B ---
wavenumbers_29, intensities_29 = load_sample("Sample 29.txt")


# Sample 0
#x0, y0, y0_std = fit_gpr(wavenumbers_0, intensities_0, label="Sample 0", color="blue")

# Sample 3B
x3b, y3b, y3b_std = fit_gpr(wavenumbers_3b, intensities_3b, label="Sample 3B", color="green")

#Sample 29
x29, y29, y29_std = fit_gpr(wavenumbers_29, intensities_29, label="Sample 29", color="red")

fig, ax1 = plt.subplots(figsize=(12, 6))

# Left y-axis for Sample 29
ax1.plot(x29.ravel(), y29, color="red", label="Sample 29 – GPR Mean")
ax1.fill_between(x29.ravel(), y29 - 2*y29_std, y29 + 2*y29_std,
                 color="red", alpha=0.2, label="95% CI – Sample 0")
ax1.set_ylabel("Intensity (Sample 29)", color="red")
ax1.tick_params(axis='y', labelcolor='red')

# Twin y-axis for Sample 3B
ax2 = ax1.twinx()
ax2.plot(x3b.ravel(), y3b, color="green", label="Sample 3B – GPR Mean")
ax2.fill_between(x3b.ravel(), y3b - 2*y3b_std, y3b + 2*y3b_std,
                 color="green", alpha=0.2, label="95% CI – Sample 3B")
ax2.set_ylabel("Intensity (Sample 3B)", color="green")
ax2.tick_params(axis='y', labelcolor='green')

# Shared X-axis
ax1.set_xlabel("Raman Shift (cm⁻¹)")
ax1.set_title("GPR Smoothed Spectra: Sample 29 vs Sample 3B (Dual Y-Axis)")
ax1.grid(True)

plt.tight_layout()
plt.savefig("sample29_vs_3b_dual_y.png", dpi=300)
#plt.show()