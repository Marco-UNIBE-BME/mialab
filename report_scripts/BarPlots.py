import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime  # Import datetime for dynamic filenames

# Load the first dataset (Dices values)
data_dices = pd.read_csv('DataFiles/Opening&closing_experiments_validation_set_DICE_with_std.csv')

# Load the second dataset (Hausdorff Distances)
data_hd = pd.read_csv('DataFiles/Opening&closing_experiments_validation_set_HDRFDST_with_std.csv')

# Extract metrics dynamically for both datasets
metrics_hd = [col.split('_')[0] for col in data_hd.columns if '_mean' in col]
metrics_dices = [col.split('_')[0] for col in data_dices.columns if '_mean' in col]
kernels_hd = data_hd['Kernel']
kernels_dices = data_dices['Kernel']

# Prepare data for plotting
def prepare_data(data, metrics):
    means = data[[f"{metric}_mean" for metric in metrics]]
    stds = data[[f"{metric}_std" for metric in metrics]]
    return means, stds

means_hd, stds_hd = prepare_data(data_hd, metrics_hd)
means_dices, stds_dices = prepare_data(data_dices, metrics_dices)

# Set up plot parameters
bar_width = 0.15  # Width of each bar
x_positions_hd = np.arange(len(kernels_hd))  # X positions for Hausdorff Distances
x_positions_dices = np.arange(len(kernels_dices))  # X positions for Dices values

# Create side-by-side subplots with slightly larger figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# Plot for Dices values (left plot)
for i, metric in enumerate(metrics_dices):
    axs[0].bar(
        x_positions_dices + i * bar_width,  # Offset for each metric's bar
        means_dices[f"{metric}_mean"],  # Heights of bars
        bar_width,
        yerr=stds_dices[f"{metric}_std"],  # Error bars
        label=metric,
        capsize=5  # Adds caps to error bars
    )

axs[0].set_title("Mean Dice Coefficients", fontsize=18)
axs[0].set_xlabel("Post-processing Method", fontsize=15)
axs[0].set_ylabel("Dice Coefficient [-]", fontsize=15)
axs[0].set_ylim(0, 1)  # Limit y-axis to 1
axs[0].set_yticks(np.arange(0, 1.1, 0.2))  # Set y-ticks at intervals of 0.2
axs[0].set_xticks(x_positions_dices + (len(metrics_dices) - 1) * bar_width / 2)
axs[0].set_xticklabels(kernels_dices, fontsize=12)  # Keep x-axis labels unrotated
axs[0].tick_params(axis='y', labelsize=12)
axs[0].legend(title="Region", title_fontsize=12, fontsize=10)
axs[0].grid(axis='y', linestyle='--', alpha=0.6)

# Plot for Hausdorff Distances (right plot)
for i, metric in enumerate(metrics_hd):
    axs[1].bar(
        x_positions_hd + i * bar_width,  # Offset for each metric's bar
        means_hd[f"{metric}_mean"],  # Heights of bars
        bar_width,
        yerr=stds_hd[f"{metric}_std"],  # Error bars
        label=metric,
        capsize=5  # Adds caps to error bars
    )

axs[1].set_title("Mean 95% Hausdorff Distances", fontsize=18)
axs[1].set_xlabel("Post-processing Method", fontsize=15)
axs[1].set_ylabel("95% Hausdorff Distance [mm]", fontsize=15)
axs[1].set_ylim(0, max(means_hd.max()) * 1.3)  # Extend y-axis slightly
axs[1].set_xticks(x_positions_hd + (len(metrics_hd) - 1) * bar_width / 2)
axs[1].set_xticklabels(kernels_hd, fontsize=12)  # Keep x-axis labels unrotated
axs[1].tick_params(axis='y', labelsize=12)
axs[1].legend(title="Region", title_fontsize=12, fontsize=10)
axs[1].grid(axis='y', linestyle='--', alpha=0.6)

# Generate dynamic filename with date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
pdf_filename = f"bar_plot_{current_time}.pdf"

# Adjust layout and save as PDF
plt.tight_layout()
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
print(f"Figure saved as {pdf_filename}")
plt.show()
