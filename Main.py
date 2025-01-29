import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import Functions

"""
Create Correlation Matrices
"""
# Define parameters
fisher = True  # option to choose if the Fisher transform will be applied to the data

# Check datatype for 'fisher' variable
assert type(fisher) == bool, "'Fisher parameter is not a boolean. Please assign a logical value (1 if You want Fisher transform to be applied)'"

# Define data to work on
paradigm = "story"    # option to choose the paradigm. "story" for Story Comprehension and "resting" for Resting State

# Story Comprehension
if (paradigm == "story"):
    language_data_path = os.path.join("Data", "Alice_TimeSeriesData_Language_StoryComprehension")
    md_data_path = os.path.join("Data", "Alice_TimeSeriesData_MD_StoryComprehension")

#Resting State
elif (paradigm == "resting"):
    language_data_path = os.path.join("Data", "Alice_TimeSeriesData_Language_StoryComprehension")
    md_data_path = os.path.join("Data", "Alice_TimeSeriesData_MD_StoryComprehension")

# Check paradigm value
if paradigm not in ["resting", "story"]:
    raise ValueError("Unexpected paradigm selected. Please choose 'story' for story comprehension or 'resting' for Resting State.")
# Check if paths exist
if not os.path.exists(language_data_path):
    raise FileNotFoundError(f"Path to the Language Data folder does not exist: {language_data_path}")
if not os.path.exists(md_data_path):
    raise FileNotFoundError(f"Path to the MD Data folder does not exist: {md_data_path}")

#Output path for saving results
output_folder = r"Figures"

# Ensure the output folder exists, create it if not
try:
    os.makedirs(output_folder, exist_ok=True)
except Exception as e:
    raise OSError(f"Failed to create output directory {output_folder}: {e}")


# Create a list of languages to be processed
language_list = [
    "Armenian", "Irish", "Greek", "Catalan", "French", "Italian", "Portuguese", "Romanian", "Spanish", "Afrikaans",
    "Danish", "Dutch", "English", "German", "Norwegian", "Swedish", "Belarusian", "Bulgarian", "Czech", "Latvian",
    "Lithuanian", "Polish", "Russian", "Serbocroatian", "Slovene", "Ukrainian", "Farsi", "Gujarati", "Hindi",
    "Marathi", "Nepali", "Arabic", "Hebrew", "Vietnamese", "Tagalog", "Tamil", "Telugu", "Japanese", "Korean", "Swahili",
    "Mandarin", "Finnish", "Hungarian", "Turkish", "Basque"
]

# Prepare a dictionary to store region averages
region_averages = {}

# Dictionary to store matrix values for each language
all_matrices = {}

# Loop through each language
for target_language in language_list:

    print(f"Processing language: {target_language}")

    try:
        # Process the data files for the current language
        language_data, md_data = Functions.process_language_files(
            language=target_language,
            language_folder=language_data_path,
            md_folder=md_data_path
        )

         #Check if data was successfully processed
        if language_data is None or md_data is None:
            raise ValueError(f"Failed to process data for {target_language}. Skipping.")

        # Debugging, checking shape of the resultant data
        print(f"Language data shape: {language_data.shape}")
        print(f"MD data shape: {md_data.shape}")

        # Combine Language and MD matrices
        combined_matrix = np.vstack([language_data, md_data])

        # Debugging, shape of the combined matrix
        print(f"Combined matrix shape: {combined_matrix.shape}")

        # Compute the full 30x30 correlation matrix
        full_corr_matrix = np.corrcoef(combined_matrix)

        # Debugging, shape of the full correlation matrix
        print(f"Full correlation matrix shape: {full_corr_matrix.shape}")

        # Calculate region averages and store them
        if (fisher == True):

            # Apply Fisher transformation
            fisher_corr_matrix = Functions.fisher_transform(full_corr_matrix)

            # Store the currently processed correlation matrix
            all_matrices[target_language] = fisher_corr_matrix

            # Calculate region averages
            averages = Functions.calculate_region_averages(fisher_corr_matrix)

        else:

            all_matrices[target_language] = full_corr_matrix

            # Calculate region averages 
            averages = Functions.calculate_region_averages(full_corr_matrix)

        #Check if averages were successfully calculated
        if averages is None:
            raise ValueError(f"Failed to calculate averages for {target_language}. Skipping.")

        # Save the region averages for currently processed language
        region_averages[target_language] = {
            "Language_avg": averages[0],
            "MD_avg": averages[1],
            "Lang_MD_avg": averages[2],
        }

        # Visualize and save the combined matrix
        if (fisher == True):
            Functions.visualize_and_save_matrix(fisher_corr_matrix, target_language, fisher, paradigm, output_folder)
        else:
            Functions.visualize_and_save_matrix(full_corr_matrix, target_language, fisher, paradigm, output_folder)

    except Exception as e:
        print(f"An error occurred while processing {target_language}: {e}")

# Debugging. Checking shapes and types for correlation matrices for all languages
for lang, matrix in all_matrices.items():
    print(f"Language: {lang}, Matrix shape: {matrix.shape}, Type: {type(matrix)}")

# Extract all matrices and compute the average
matrix_list = list(all_matrices.values())
average_matrix = np.mean(matrix_list, axis=0)  # Element-wise average
print(f"Averaged matrix shape: {average_matrix.shape}")

# Plot the Averaged Matrix and save it
Functions.visualize_and_save_matrix(average_matrix, "average", fisher, paradigm, output_folder)

"""
Create the Average Correlation Boxplots
"""

# Define the keys for the categories
categories = ["Language_avg", "Lang_MD_avg", "MD_avg"]

# Extract data for each category
data = {category: [region_averages[lang][category] for lang in region_averages] for category in categories}

# Calculate boxplot components for each category
boxplot_components = Functions.calculate_boxplot_components(data)

# Print the results
for category, stats in boxplot_components.items():
    print(f"Category: {category}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

# Plot and save boxplot
Functions.plot_custom_boxplot(boxplot_components, ["Language_avg", "Lang_MD_avg", "MD_avg"], data, fisher, paradigm, output_folder)

"""
Create Integration vs Dissociation plot
"""

# Convert region_averages to a DataFrame
results_df = pd.DataFrame.from_dict(region_averages, orient='index').reset_index()
results_df.columns = ['Language', 'Integration_Language', 'Integration_MD', 'Dissociation']

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(
    results_df['Integration_Language'], 
    results_df['Dissociation'], 
    alpha=0.7, 
    label='Languages'
)

# Trendline
slope, intercept, r_value, p_value, std_err = linregress(results_df['Integration_Language'], results_df['Dissociation'])
x_vals = np.linspace(results_df['Integration_Language'].min(), results_df['Integration_Language'].max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Trendline (r={r_value:.2f}, p={p_value:.4f})')

# Plot settings
if (fisher == True):
    if (paradigm == "story"):
        plt.title("Integration and Dissociation during Story Comprehension (Fisher Transform)")
    else:
        plt.title("Integration and Dissociation during Rest (Fisher Transform)")
elif (paradigm == "story"):
        plt.title("Integration and Dissociation during Story Comprehension (No Fisher Transform)")
else:
        plt.title("Integration and Dissociation during Rest (No Fisher Transform)")

plt.xlabel("Integration (Intra-Network Correlation)")
plt.ylabel("Dissociation (Inter-Network Correlation)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the figure
output_path = os.path.join(output_folder, r"Fig11.png")
plt.savefig(output_path)

plt.show()

# Print statistical analysis
print(f"Pearson's r: {r_value:.2f}, p-value: {p_value:.4f}")