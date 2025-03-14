import os
from config.config_handler import read_config, create_config
from data_processing.file_processing import cycle_through_languages
from visualization.boxplot import calculate_boxplot_components, plot_custom_boxplot
from visualization.integration_plot import plot_integration_vs_dissociation
from logger import logger

"""
Configuration
"""
# Making sure config file exists
config_file_path = "config.ini"
if not os.path.exists(config_file_path):
    logger.info("Configuration file not found. Creating a default config file...")
    create_config()

# Load parameters from config file
config_values = read_config()

# Assign parameter values
fisher = config_values['fisher']  # Option to choose if the Fisher transform will be applied
paradigm = config_values['paradigm']  # "story" for Story Comprehension, "resting" for Resting State
output_folder = config_values['output_folder']  # Output path for saving results
language_list = config_values['language_list']  # List of languages to be processed

# Validate 'fisher' variable type
if not isinstance(fisher, bool):
    logger.error("Invalid type for 'fisher': Expected bool, got %s", type(fisher))
    raise TypeError("'Fisher' parameter is not a boolean. Please assign True/False.")

# Define data paths based on paradigm
if paradigm == "story":
    language_data_path = os.path.join("Data", "Alice_TimeSeriesData_Language_StoryComprehension")
    md_data_path = os.path.join("Data", "Alice_TimeSeriesData_MD_StoryComprehension")
elif paradigm == "resting":
    language_data_path = os.path.join("Data", "Alice_TimeSeriesData_Language_RestingState")
    md_data_path = os.path.join("Data", "Alice_TimeSeriesData_MD_RestingState")
else:
    logger.error("Invalid value for 'paradigm': Expected 'story' or 'resting', got %s", paradigm)
    raise ValueError("Unexpected paradigm selected. Choose 'story' or 'resting'.")

# Check if data paths exist
if not os.path.exists(language_data_path):
    logger.error("Missing data folder: Path to the Language Data folder does not exist:  %s", language_data_path)
    raise FileNotFoundError(f"Path to the Language Data folder does not exist: {language_data_path}")
if not os.path.exists(md_data_path):
    logger.error("Missing data folder: Path to the MD Data folder does not exist: %s", language_data_path)
    raise FileNotFoundError(f"Path to the MD Data folder does not exist: {md_data_path}")

# Ensure the output folder exists
try:
    os.makedirs(output_folder, exist_ok=True)
except Exception as e:
    logger.error("Failed folder creation: Unable to create this output directory:  %s", output_folder)
    raise OSError(f"Failed to create output directory {output_folder}: {e}")

"""
Create Correlation Matrices
"""
# Prepare a dictionary to store region averages
region_averages: dict[str, dict[str, float]] = {}

# Process languages and create correlation matrices
cycle_through_languages(language_list, language_data_path, md_data_path, region_averages, fisher, paradigm, output_folder)

"""
Create the Average Correlation Boxplots
"""
# Define categories
categories = ["Language_avg", "Lang_MD_avg", "MD_avg"]

# Extract data for each category
data = {category: [region_averages[lang][category] for lang in region_averages] for category in categories}

# Calculate boxplot components
boxplot_components = calculate_boxplot_components(data)

# Print results
for category, stats in boxplot_components.items():
    logger.info("Category: %s", category)
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)
    print()

# Generate and save boxplot
plot_custom_boxplot(boxplot_components, categories, data, fisher, paradigm, output_folder)

"""
Create Integration vs Dissociation Plot
"""
plot_integration_vs_dissociation(region_averages, fisher, paradigm, output_folder)