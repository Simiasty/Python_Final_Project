import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

def process_language_files(language, language_folder, md_folder):

    # Initialize variables for language and md data
    language_data = []
    md_data = []

    # Process files in the Language folder
    for file in os.listdir(language_folder):
        if file.endswith('.csv') and language in file:  # Match the language

            file_path = os.path.join(language_folder, file) # Read the path of an individual file
            df = pd.read_csv(file_path)  # Load CSV file
            df = df.fillna(0)   # Fill missing values with 0s

            # Extract only ROI activation columns (starting from 5th column)
            language_data.append(df.iloc[:, 4:].values)

    # Process files in the MD folder (same as Language)
    for file in os.listdir(md_folder):
        if file.endswith('.csv') and language in file:

            file_path = os.path.join(md_folder, file)
            df = pd.read_csv(file_path)
            df = df.fillna(0)

            md_data.append(df.iloc[:, 4:].values)

    # Average across participants for both systems
    language_matrix = np.mean(np.stack(language_data, axis=0), axis=0)  # Average for Language system
    md_matrix = np.mean(np.stack(md_data, axis=0), axis=0)  # Average for MD system

    return language_matrix, md_matrix

# Function for performing the Fisher Transform on the data
def fisher_transform(correlation_matrix):
    # Clip values to avoid division by zero in arctanh
    clipped_matrix = np.clip(correlation_matrix, -0.9999, 0.9999)
    return np.arctanh(clipped_matrix)

# Function for calculating region averages for a matrix
def calculate_region_averages(matrix):

    # Create a mask to exclude the main diagonal
    mask = np.eye(matrix.shape[0], dtype=bool)
    
    # Define the specific regions
    Language_region = matrix[:12, :12]
    MD_region = matrix[12:, 12:]
    
    # Lang_MD is the rest of the matrix (upper-right and lower-left)
    Lang_MD_top_right = matrix[:12, 12:]
    Lang_MD_bottom_left = matrix[12:, :12]
    Lang_MD_region = np.concatenate((Lang_MD_top_right.flatten(), Lang_MD_bottom_left.flatten()))

    # Remove diagonal elements from Language_region and MD_region
    Language_mask = ~mask[:12, :12]
    MD_mask = ~mask[12:, 12:]
    Language_region_no_diag = Language_region[Language_mask]
    MD_region_no_diag = MD_region[MD_mask]

    # Debugging: Print values and shapes for verification
    print("Language_region shape:", Language_region_no_diag.shape)
    print("Language_region values:\n", Language_region_no_diag)
    print("MD_region shape:", MD_region_no_diag.shape)
    print("MD_region values:\n", MD_region_no_diag)
    print("Lang_MD_region size:", Lang_MD_region.size)
    print("Lang_MD_region values:\n", Lang_MD_region)

    # Calculate averages ignoring the diagonal
    Language_avg = np.mean(Language_region_no_diag)
    MD_avg = np.mean(MD_region_no_diag)
    Lang_MD_avg = np.mean(Lang_MD_region)

    # Debugging: Print calculated averages
    print("Language_avg:", Language_avg)
    print("MD_avg:", MD_avg)
    print("Lang_MD_avg:", Lang_MD_avg)

    return Language_avg, MD_avg, Lang_MD_avg

# Function to calculate boxplot components
def calculate_boxplot_components(data):
    components = {}
    for category, values in data.items():
        values = np.array(values)
        mean = np.mean(values)
        median = np.median(values)
        q1 = np.percentile(values, 25)  # 25th percentile
        q3 = np.percentile(values, 75)  # 75th percentile
        iqr = q3 - q1  # Interquartile range
        lower_whisker = max(values[values >= q1 - 1.5 * iqr].min(), q1 - 1.5 * iqr)
        upper_whisker = min(values[values <= q3 + 1.5 * iqr].max(), q3 + 1.5 * iqr)
        outliers = values[(values < lower_whisker) | (values > upper_whisker)]

        components[category] = {
            "mean": mean,
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_whisker": lower_whisker,
            "upper_whisker": upper_whisker,
            "outliers": outliers.tolist(),
        }
    return components

# Function for plotting a boxplot based on calculated parameters
def plot_custom_boxplot(components, category_labels, fisher, paradigm):

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract boxplot components in order
    medians = [components[cat]['median'] for cat in category_labels]
    q1_values = [components[cat]['q1'] for cat in category_labels]
    q3_values = [components[cat]['q3'] for cat in category_labels]
    lower_whiskers = [components[cat]['lower_whisker'] for cat in category_labels]
    upper_whiskers = [components[cat]['upper_whisker'] for cat in category_labels]
    outliers = [components[cat]['outliers'] for cat in category_labels]

    # Prepare positions for boxplots
    positions = range(len(category_labels))

    # Draw boxes, whiskers, and medians
    for i, (median, q1, q3, low, high, outs) in enumerate(zip(medians, q1_values, q3_values, lower_whiskers, upper_whiskers, outliers)):
        # Box
        ax.fill_betweenx([q1, q3], i - 0.4, i + 0.4, color='lightgray', edgecolor='black', alpha=0.8)
        # Whiskers
        ax.plot([i, i], [low, q1], color='black', linestyle='-', lw=1)
        ax.plot([i, i], [q3, high], color='black', linestyle='-', lw=1)
        # Median
        ax.plot([i - 0.4, i + 0.4], [median, median], color='black', lw=2)
        # Outliers
        ax.scatter([i] * len(outs), outs, color='black', s=10, zorder=3)
        # Title
        if (fisher == True):
            if (paradigm == "story"):
                ax.set_title("Average correlation - Story Comprehension (Fisher transform)")
            else:
                ax.set_title("Average correlation - Resting State (Fisher transform)")
        elif (paradigm == "story"):
            ax.set_title("Average correlation - Story Comprehension (No Fisher transform)")
        else:
            ax.set_title("Average correlation - Resting State (No Fisher transform)")

    # Plot individual data points
    for i, category in enumerate(category_labels):
        points = data[category]  # Get raw data points for the category
        ax.scatter([i] * len(points), points, color='black', alpha=0.2, s=15, zorder=1)

    # Set x-ticks and labels
    ax.set_xticks(positions)
    ax.set_xticklabels(category_labels)
    ax.set_ylabel('Average correlation')
    ax.axhline(y=0, color='gray', linestyle='--', lw=1)  # Add horizontal line at y=0

    plt.show()