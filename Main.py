import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from Functions import process_language_files
import Functions

# Define data to work on

paradigm = "story"    # option to choose the paradigm. "story" for Story Comprehension and "resting" for Resting State

# Story Comprehension
if (paradigm == "story"):
    language_data_path = r"C:\Users\samue\Desktop\Python_BarIlan\Python_Final_Project\Data\Alice_TimeSeriesData_Language_StoryComprehension"
    md_data_path = r"C:\Users\samue\Desktop\Python_BarIlan\Python_Final_Project\Data\Alice_TimeSeriesData_MD_StoryComprehension"

#Resting State
elif (paradigm == "resting"):
    language_data_path = r"C:\Users\samue\Desktop\Python_BarIlan\Python_Final_Project\Data\Alice_TimeSeriesData_Language_RestingState"
    md_data_path = r"C:\Users\samue\Desktop\Python_BarIlan\Python_Final_Project\Data\Alice_TimeSeriesData_MD_RestingState"

assert paradigm == "resting" or paradigm == "story", "Unexpected paradigm selected. Please choose 'story' for story comprehension or 'resting' for Resting State"
assert os.path.exists(language_data_path) == True, "Path to the Language Data folder does not exist. Please specify correct path."
assert os.path.exists(md_data_path) == True, "Path to the MD Data folder does not exist. Please specify correct path."

#Output path for saving results
output_folder = r"C:\Users\samue\Desktop\Python_BarIlan\Python_Final_Project\Matrices\Resting_NoFish"

fisher = True # option to choose if the Fisher transform will be applied to the data. If True, then transform is applied
assert type(fisher) == bool, "'Fisher parameter is not a boolean. Please assign a logical value (1 if You want Fisher transform to be applied)'"

# Ensure the output folder exists, create it if not
os.makedirs(output_folder, exist_ok=True)

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

        # Debugging, checking shape of the resultant data
        print(f"Language data shape: {language_data.shape}")
        print(f"MD data shape: {md_data.shape}")

        # Combine Language and MD matrices
        combined_matrix = np.vstack([language_data, md_data])

        # Debugging, shape of the combined matrix
        print(f"MD data shape: {md_data.shape}")

        # Compute the full 30x30 correlation matrix
        full_corr_matrix = np.corrcoef(combined_matrix)

        # Debugging, shape of the full correlation matrix
        print(f"Full correlation matrix shape: {full_corr_matrix.shape}")

        # Calculate region averages and store them
        if (fisher == True):

            # Apply Fisher transformation
            fisher_corr_matrix = fisher_transform(full_corr_matrix)

            # Store the currently processed correlation matrix
            all_matrices[target_language] = fisher_corr_matrix

            # Debugging. Verify type and shape of the currently processed matrix after saving
            print(f"Assigning matrix for {target_language}.")
            print(f"Matrix type: {type(all_matrices[target_language])}, shape: {all_matrices[target_language].shape}")

            # Calculate region averages 
            averages = calculate_region_averages(fisher_corr_matrix)

        else:

            all_matrices[target_language] = full_corr_matrix

            print(f"Assigning matrix for {target_language}.")
            print(f"Matrix type: {type(all_matrices[target_language])}, shape: {all_matrices[target_language].shape}")

            averages = calculate_region_averages(full_corr_matrix)

        # Save the region averages for currently processed language
        region_averages[target_language] = {
            "Language_avg": averages[0],
            "MD_avg": averages[1],
            "Lang_MD_avg": averages[2],
        }

        # Visualize the combined matrix
        plt.figure(figsize=(12, 12))
        if (fisher == True):
            plt.imshow(
                fisher_corr_matrix,
                cmap=plt.get_cmap('Spectral').reversed(),
                aspect='equal'
            )
        else:
           plt.imshow(
                full_corr_matrix,
                cmap=plt.get_cmap('Spectral').reversed(),
                aspect='equal'
            )

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
        output_path = os.path.join(output_folder, f"{target_language}.png")
        #plt.savefig(output_path)
        plt.close()

        print(f"Saved correlation matrix for {target_language} to {output_path}")



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
plt.figure(figsize=(12, 12))
       
plt.imshow(
    average_matrix,
    cmap=plt.get_cmap('Spectral').reversed(),
    aspect='equal')

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
output_path = os.path.join(output_folder, file_name)
plt.savefig(output_path)
plt.close()
print(f"Saved correlation matrix for Average Matrix to {output_path}")

# Define the keys for the categories
categories = ["Language_avg", "Lang_MD_avg", "MD_avg"]

# Extract data for each category
data = {category: [region_averages[lang][category] for lang in region_averages] for category in categories}

# Calculate boxplot components for each category
boxplot_components = calculate_boxplot_components(data)

# Print the results
for category, stats in boxplot_components.items():
    print(f"Category: {category}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

plot_custom_boxplot(boxplot_components, ["Language_avg", "Lang_MD_avg", "MD_avg"])

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
plt.show()

# Print statistical analysis
print(f"Pearson's r: {r_value:.2f}, p-value: {p_value:.4f}")