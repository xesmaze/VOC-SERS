import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import colormaps as cmx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Folder containing the sample files
sample_folder = "sample_data"

# Find all matching files
sample_paths = glob.glob(os.path.join(sample_folder, "Sample*.txt"))

# Sort to ensure consistent order (optional)
sample_paths.sort()

# Dictionary to hold loaded data
samples = {}

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
    plt.savefig(f"{label}", dpi=300)
    plt.close()
    return x_pred, y_pred, y_std

# Load samples one at a time
# --- Sample 0 - Blank paper disk with no AgNP treatment---
#wavenumbers_0, intensities_0 = load_sample("Sample 0.txt")

# --- Sample 3B ---
#wavenumbers_3b, intensities_3b = load_sample("Sample 3B.txt")

# --- Sample 3B ---
#wavenumbers_29, intensities_29 = load_sample("Sample 29.txt")

# Load all samples
# Find all sample files
sample_files = sorted(glob.glob(os.path.join(sample_folder, "Sample*.txt")))
samples = {}
for filepath in sample_files:
    label = os.path.splitext(os.path.basename(filepath))[0]  # e.g., 'Sample 0'
    wavenumbers, intensities = load_sample(filepath)
    samples[label] = {
        "wavenumbers": wavenumbers,
        "intensities": intensities
    }

# Fit and store GPR for Sample 0 to be used as reference
ref_label = "Sample 0"
ref_data = samples[ref_label]
x0, y0, y0_std = fit_gpr(ref_data["wavenumbers"], ref_data["intensities"], label=ref_label, color="blue")

# Visualization
# Create a color dictionary (excluding the reference)
nonref_labels = [label for label in samples.keys() if label != ref_label]
n_colors = len(nonref_labels)

# Pick colors from a colormap (e.g., 'tab10', 'Set3', or 'viridis')
# Use resampled colormap
colormap = cmx.get_cmap("tab10").resampled(n_colors)
color_dict = {
    label: mcolors.to_hex(colormap(i)) for i, label in enumerate(nonref_labels)
}

# Compare each other sample to Sample 0
for label, data in samples.items():
    if label == ref_label:
        continue  # skip comparison to itself
    # Fit GPR
    x, y, y_std = fit_gpr(data["wavenumbers"], data["intensities"], label=label, color="blue")

    # Single plot with Dual y-axis to compare intensities
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis
    ax1.plot(x0.ravel(), y0, color="black", label=f"{ref_label} – GPR Mean")
    ax1.fill_between(x0.ravel(), y0 - 2 * y0_std, y0 + 2 * y0_std,
                     color="black", alpha=0.2, label="95% CI – Sample 0")
    ax1.set_ylabel(f"Intensity {ref_label}", color="black")
    ax1.tick_params(axis='y', labelcolor='black')

    # Twin y-axis for Sample to compare
    ax2 = ax1.twinx()
    ax2.plot(x.ravel(), y, color=color_dict[label], label=f"GPR mean – {label}")
    ax2.fill_between(x.ravel(), y - 2 * y_std, y + 2 * y_std, color=color_dict[label], alpha=0.2, label=f"95% CI – {label}")
    ax2.set_ylabel(f"Intensity ({label})", color=color_dict[label])
    ax2.tick_params(axis='y', labelcolor=color_dict[label])

    # Shared X-axis and Title
    ax1.set_xlabel("Raman Shift (cm⁻¹)")
    ax1.set_title(f"GPR Smoothed Spectra: {ref_label} vs {label} (Dual Y-Axis)")
    ax1.grid(True)

    # Save
    plt.tight_layout()
    plt.savefig(f"{ref_label}_vs_{label.replace(' ', '_')}_dual_y.png", dpi=300)
    plt.close()

    # Stacked Subpanels
    fig2, (ax3, ax4) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(12, 8), height_ratios=[1, 1]
    )
    # Subplot 1 : # Top: Ref
    ax3.plot(x0.ravel(), y0, color="black", label=f"GPR mean – {ref_label}")
    ax3.fill_between(x0.ravel(), y0 - 2*y0_std, y0 + 2*y0_std, color="black", alpha=0.2, label=f"95% CI – {ref_label}")
    ax3.set_ylabel("Intensity (a.u.)")
    ax3.set_title(f"GPR Smoothed Spectra: {ref_label}")
    ax3.legend()
    ax3.grid(True)

    ax4.plot(x.ravel(), y, color="blue", label=f"GPR mean – {label}")
    plt.fill_between(x.ravel(), y - 2*y_std, y + 2*y_std, color="blue", alpha=0.2, label=f"95% CI – {label}")
    ax4.set_ylabel("Intensity (a.u.)")
    ax4.set_title(f"GPR Smoothed Spectra: {label}")
    ax4.legend()
    ax4.grid(True)


    # Final plot settings
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"GPR Comparison: {label} vs {ref_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save
    plt.tight_layout()
    outname = f"Stacked_{label.replace(' ', '_')}_vs_{ref_label.replace(' ', '_')}.png"
    plt.savefig(outname, dpi=300)
    plt.close()

# fig, ax1 = plt.subplots(figsize=(12, 6))
#
# # Left y-axis for Sample 29
# ax1.plot(x29.ravel(), y29, color="red", label="Sample 29 – GPR Mean")
# ax1.fill_between(x29.ravel(), y29 - 2*y29_std, y29 + 2*y29_std,
#                  color="red", alpha=0.2, label="95% CI – Sample 0")
# ax1.set_ylabel("Intensity (Sample 29)", color="red")
# ax1.tick_params(axis='y', labelcolor='red')
#
# # Twin y-axis for Sample 3B
# ax2 = ax1.twinx()
# ax2.plot(x3b.ravel(), y3b, color="green", label="Sample 3B – GPR Mean")
# ax2.fill_between(x3b.ravel(), y3b - 2*y3b_std, y3b + 2*y3b_std,
#                  color="green", alpha=0.2, label="95% CI – Sample 3B")
# ax2.set_ylabel("Intensity (Sample 3B)", color="green")
# ax2.tick_params(axis='y', labelcolor='green')
#
# # Shared X-axis
# ax1.set_xlabel("Raman Shift (cm⁻¹)")
# ax1.set_title("GPR Smoothed Spectra: Sample 29 vs Sample 3B (Dual Y-Axis)")
# ax1.grid(True)
#
# plt.tight_layout()
# plt.savefig("sample29_vs_3b_dual_y.png", dpi=300)
# #plt.show()