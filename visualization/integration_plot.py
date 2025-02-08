import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress
from logger import logger

def plot_integration_vs_dissociation(region_averages, fisher, paradigm, output_folder):
    """
    Plots the Integration vs Dissociation plot and saves it as a .png file.

    Inputs:
        - region_averages (dict): Dictionary containing average values for each of the three regions for each language
        - fisher (bool): Whether the Fisher transform was applied
        - paradigm (str): Specifies if the paradigm is 'story' or 'rest'
        - output_folder (str): Path to the output directory
    """
    if not os.path.isdir(output_folder):
        logger.error("Missing folder: Path to the output folder does not exist:  %s", output_folder)
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    # Convert region_averages to a DataFrame
    results_df = pd.DataFrame.from_dict(region_averages, orient='index').reset_index()
    results_df.columns = ['Language', 'Integration_Language', 'Integration_MD', 'Dissociation']

    # Scatter plot of Integration (x-axis) vs Dissociation (y-axis)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        results_df['Integration_Language'], 
        results_df['Dissociation'], 
        alpha=0.7, 
        label='Languages'
    )

    # Linear regression (trendline)
    slope, intercept, r_value, p_value, std_err = linregress(
        results_df['Integration_Language'], results_df['Dissociation']
    )
    x_vals = np.linspace(results_df['Integration_Language'].min(), results_df['Integration_Language'].max(), 100)
    y_vals = slope * x_vals + intercept

    plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Trendline (r={r_value:.2f}, p={p_value:.4f})')

    # Title settings based on Fisher transform and paradigm
    if fisher:
        title_prefix = "Integration and Dissociation (Fisher Transform)"
    else:
        title_prefix = "Integration and Dissociation (No Fisher Transform)"

    title_suffix = "Story Comprehension" if paradigm == "story" else "Resting State"
    plt.title(f"{title_prefix} during {title_suffix}")

    plt.xlabel("Integration (Intra-Network Correlation)")
    plt.ylabel("Dissociation (Inter-Network Correlation)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_folder, "Fig11.png")
    plt.savefig(output_path)
    plt.show()

    logger.info("Saved Integration vs. Dissociation plot to: %s", output_path)
