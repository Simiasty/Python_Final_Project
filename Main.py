import os


import Functions

"""
Configuration
"""
# Load parameters from config file
config_values = Functions.read_config()

# Assign parameter values
fisher = config_values['fisher'] # option to choose if the Fisher transform will be applied to the data
paradigm = config_values['paradigm'] # option to choose the paradigm. "story" for Story Comprehension and "resting" for Resting State
output_folder = config_values['output_folder'] # Output path for saving results
language_list = config_values['language_list'] # List of languages to be processed

# Check datatype for 'fisher' variable
assert type(fisher) is bool, "'Fisher parameter is not a boolean. Please assign a logical value (1 if You want Fisher transform to be applied)'"  

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

# Ensure the output folder exists, create it if not
try:
    os.makedirs(output_folder, exist_ok=True)
except Exception as e:
    raise OSError(f"Failed to create output directory {output_folder}: {e}")


"""
Create Correlation Matrices
"""
# Prepare a dictionary to store region averages
region_averages = {}

# Go through all the languages in the list
Functions.cycle_through_languages(language_list, language_data_path, md_data_path, region_averages, fisher, paradigm, output_folder)

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
Functions.plot_integration_vs_dissociation(region_averages, fisher, paradigm, output_folder)