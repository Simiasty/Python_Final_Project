import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

def process_language_files(language, language_folder, md_folder):
    """
    Processes .csv files stored in the specified paths into matrices.

    Inputs:
        - language (str), name of the language for which files are to be processed
        - language_folder (str), path to the directory containing .csv files with data concerning the Language network
        - md_folder (str), path to the directory containing .csv files with data concerning the MD network

    Outputs:
        - language_matrix (np array, 12x142 or 12x150), matrix containing the numerical data for the Language network
        - md_matrix (np array, 18x142 or 18x150), matrix containing the numerical data for the MD network

    """

    # Initialize variables for language and md data
    language_data = []
    md_data = []

    # Check if the folders exist
    if not os.path.exists(language_folder):
        raise FileNotFoundError(f"Language folder not found: {language_folder}")
    if not os.path.exists(md_folder):
        raise FileNotFoundError(f"MD folder not found: {md_folder}")

    # Process files in the Language folder
    for file in os.listdir(language_folder):
        if file.endswith('.csv') and language in file:  # Match the language

            file_path = os.path.join(language_folder, file) # Read the path of an individual file
            try:  
                df = pd.read_csv(file_path)  # Load CSV file
                df = df.fillna(0)   # Fill missing values with 0s

                # Extract only ROI activation columns (starting from 5th column)
                language_data.append(df.iloc[:, 4:].values)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    if not language_data: # Check if data was loaded
        raise ValueError(f"No valid language files found for {language} in {language_folder}")

    # Process files in the MD folder (same as Language)
    for file in os.listdir(md_folder):
        if file.endswith('.csv') and language in file:

            file_path = os.path.join(md_folder, file)
            try:  # Add error handling for file reading**
                df = pd.read_csv(file_path)
                df = df.fillna(0)

                md_data.append(df.iloc[:, 4:].values)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    if not md_data:  # Check if data was loaded**
        raise ValueError(f"No valid MD files found for {language} in {md_folder}")

    # Average across participants for both systems
    try: # Handle potential shape or data issues
        language_matrix = np.mean(np.stack(language_data, axis=0), axis=0)  # Average for Language system
        md_matrix = np.mean(np.stack(md_data, axis=0), axis=0)  # Average for MD system
    except Exception as e:
        raise ValueError(f"Error averaging data: {e}")

    return language_matrix, md_matrix

# Function for performing the Fisher Transform on the data
def fisher_transform(correlation_matrix):
    """
    Performs Fisher transform on the input matrix

    Input:
        - correlation matrix (np array, 30x30), matrix containing pairwise Pearson's correlations for each of the ROIs

    Output:
        - np.arctanh(clipped matrix) (np array, 30x30), input matrix transformed with Fisher transform
    """
    # Clip values to avoid division by zero in arctanh
    clipped_matrix = np.clip(correlation_matrix, -0.9999, 0.9999)
    return np.arctanh(clipped_matrix)

# Function for calculating region averages for a matrix
def calculate_region_averages(matrix):
    """
    Takes the input matrix and calculates averages values for predetermined subregions, while ignoring the diagonal.

    Inputs:
        - matrix (numpy array, 30x30)

    Output:
        - Language_avg, (float), average value for the region coresponding to the language network
        - MD_avg (float), average value for the region coresponding to the MD network
        - Language_MD_avg (float), average value for the region coresponding to the inter-network correlation regions
    """

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
    """
    Calculates components for the boxplot from figure 3c

    Input:
        - data (dict), Dictionary cantaining three vectors with names of the averaged regions 
                       containing average values for this region for each language
    Output:
        - components (dict), Dictionary of vectors containing boxplot components for the key categories in the original dictionary
    """
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
def plot_custom_boxplot(components, category_labels, data, fisher, paradigm, output_folder):
    """
    Plots the boxplot from figure 3c.

    Inputs:
        - components (dict), Dictionary of vectors containing boxplot components for the key categories in the original dictionary
        - category_labels (list), List of category labels for naming the ticks on the x-axis
        - data (dict), Dictionary cantaining three vectors with names of the averaged regions 
                       containing average values for this region for each language
        - fisher (bool), logical value used to determine if Fisher transform is to be performed. Here used for determining the plot title.
        - paradigm (str), string determining paradigm explored. Here used for determining the plot title.
    Output:
        
    """
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

    # Save the figure
    output_path = os.path.join(output_folder, r"boxplot.png")
    plt.savefig(output_path)

    plt.show()

def visualize_and_save_matrix (matrix, target_language, fisher, paradigm, output_folder):
    """
    Plots the input matrix using matplotlib and saves the result as a .png image.

    Inputs:
        - target_language (str), string containing the name of the language for which the matrix was calculated.
                                 The value can be set to 'average' to indicate processing of the matrix containing average corralations
                                 for all languages.
        - fisher (bool), logical value used to determine if Fisher transform is to be performed. Here used for determining plot and file titles.
        - paradigm (str), string determining paradigm explored. Here used for determining the plot and file titles.
        - output_folder (str), string containing the path to the output directory
        
    """
    # Visualize the combined matrix
    plt.figure(figsize=(12, 12))
    
    plt.imshow(
        matrix,
        cmap=plt.get_cmap('Spectral').reversed(),
        aspect='equal'
        )
    
    if (target_language == "average"):
        if (fisher == True):
            if (paradigm == "story"):
                plt.title(y=1.1, label="Average Correlation Matrix (Story Comprehension, Fisher Transform)", size='xx-large')
                file_name = "Avg_Cor_Story_Fisher.png"
            else:
                plt.title(y=1.1, label="Average Correlation Matrix (Resting State, Fisher Transform)", size='xx-large')
                file_name = "Avg_Cor_Rest_Fisher.png"
        elif (paradigm == "story"):
            plt.title(y=1.1, label="Average Correlation Matrix (Story Comprehension, No Fisher Transform)", size='xx-large')
            file_name = "Avg_Cor_Story_NoFisher.png"
        else:
            plt.title(y=1.1, label="Average Correlation Matrix (Resting State, No Fisher Transform)", size='xx-large')
            file_name = "Avg_Cor_Rest_NoFisher.png"
    else:
        plt.title(y=1.1, label=target_language, size='xx-large')

    # Set lines separating regions and hemispheres for better visibility
    plt.axhline(y=5.5, color='black', linewidth=1.5)
    plt.axhline(y=11.5, color='black', linewidth=2.0)
    plt.axhline(y=20.5, color='black', linewidth=1.5)

    plt.axvline(x=5.5, color='black', linewidth=1.5)
    plt.axvline(x=11.5, color='black', linewidth=2.0)
    plt.axvline(x=20.5, color='black', linewidth=1.5)

    # Add section labels for Language and MD (LH and RH)
    plt.text(-0.5, 3, "Language LH", va='center', ha='right', fontsize=12, rotation=90, color='black')
    plt.text(-0.5, 8.5, "Language RH", va='center', ha='right', fontsize=12, rotation=90, color='black')
    plt.text(-0.5, 14.5, "MD LH", va='center', ha='right', fontsize=12, rotation=90, color='black')
    plt.text(-0.5, 25, "MD RH", va='center', ha='right', fontsize=12, rotation=90, color='black')

    plt.text(3, -1.0, "Language LH", va='center', ha='center', fontsize=12, color='black')
    plt.text(8.5, -1.0, "Language RH", va='center', ha='center', fontsize=12, color='black')
    plt.text(14.5, -1.0, "MD LH", va='center', ha='center', fontsize=12, color='black')
    plt.text(25, -1.0, "MD RH", va='center', ha='center', fontsize=12, color='black')

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    
    # Save the figure
    if (target_language == "average"):
        output_path = os.path.join(output_folder, f"{file_name}")
    else:
        output_path = os.path.join(output_folder, f"{target_language}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved correlation matrix for {target_language} to {output_path}")