import os
import pandas as pd
import numpy as np

from data_processing.matrix_operations import fisher_transform, calculate_region_averages
from visualization.matrix_plot import visualize_and_save_matrix
from logger import logger


def process_files_in_folder(folder, language):
    """
    Reads all CSV files in a folder matching a specific language name.

    Inputs:
        - folder (str): Path to directory containing CSV files.
        - language (str): Name of the language to filter files.

    Returns:
        - List of NumPy arrays containing extracted data.
    """
    data = []
    if not os.path.exists(folder):
        logger.error("Folder not found: No such directory as:  %s", folder)
        raise FileNotFoundError(f"Folder not found: {folder}")

    for file in os.listdir(folder):
        if file.endswith('.csv') and language in file:
            file_path = os.path.join(folder, file)
            try:
                df = pd.read_csv(file_path)
                df = df.fillna(0)  # Replace NaN values with 0
                data.append(df.iloc[:, 4:].values)  # Extract numerical data
            except Exception as e:
                logger.error("File not found: No such file as:  %s", file_path)
                print(f"Error reading file {file_path}: {e}")
    return data

def process_language_files(language, language_folder, md_folder):
    """
    Processes .csv files stored in the specified paths into matrices.

    Inputs:
        - language (str): Language name to filter files.
        - language_folder (str): Path to the directory containing CSV files for the Language network.
        - md_folder (str): Path to the directory containing CSV files for the MD network.

    Returns:
        - language_matrix (np.ndarray): Processed matrix for the Language network.
        - md_matrix (np.ndarray): Processed matrix for the MD network.
    """
    language_data = process_files_in_folder(language_folder, language)
    md_data = process_files_in_folder(md_folder, language)

    if not language_data:
        logger.error("Data does not exist: language_data has no assigned values")
        raise ValueError(f"No valid language files found for {language} in {language_folder}")
    if not md_data:
        logger.error("Data does not exist: md_data has no assigned values")
        raise ValueError(f"No valid MD files found for {language} in {md_folder}")

    return np.mean(np.stack(language_data, axis=0), axis=0), np.mean(np.stack(md_data, axis=0), axis=0)

def cycle_through_languages(language_list, language_data_path, md_data_path, region_averages, fisher, paradigm, output_folder):
    """
    Iterates through each language, processes data, computes correlation matrices, and visualizes results.

    Inputs:
        - language_list (list): List of languages to process.
        - language_data_path (str): Path to the directory containing language data.
        - md_data_path (str): Path to the directory containing MD data.
        - region_averages (dict): Dictionary to store computed averages.
        - fisher (bool): Whether to apply Fisher transformation.
        - paradigm (str): Paradigm type ('story' or 'rest').
        - output_folder (str): Where to save visualizations.

    Outputs:
        - Updates `region_averages` dictionary with computed values.
        - Saves correlation matrices and average plots.
    """
    all_matrices = {}

    for target_language in language_list:

        logger.info("Processing language: %s", target_language)

        try:
            language_data, md_data = process_language_files(
                language=target_language,
                language_folder=language_data_path,
                md_folder=md_data_path
            )

            try:
                if language_data is None or md_data is None:
                    raise ValueError(f"Failed to process data for {target_language}. Skipping.")
            except ValueError as e:
                logger.error(e)

            combined_matrix = np.vstack([language_data, md_data])
            full_corr_matrix = np.corrcoef(combined_matrix)

            if fisher:
                fisher_corr_matrix = fisher_transform(full_corr_matrix)
                all_matrices[target_language] = fisher_corr_matrix
                averages = calculate_region_averages(fisher_corr_matrix)
            else:
                all_matrices[target_language] = full_corr_matrix
                averages = calculate_region_averages(full_corr_matrix)
            try:
                if averages is None:
                    raise ValueError(f"Failed to calculate averages for {target_language}. Skipping.")
            except ValueError as e:
                logger.error(e)

            region_averages[target_language] = {
                "Language_avg": averages[0],
                "MD_avg": averages[1],
                "Lang_MD_avg": averages[2],
            }

            visualize_and_save_matrix(all_matrices[target_language], target_language, fisher, paradigm, output_folder)

        except Exception as e:
            logger.error(f"An error occurred while processing {target_language}: {e}")

    # Compute and save the average correlation matrix across languages
    matrix_list = list(all_matrices.values())
    average_matrix = np.mean(matrix_list, axis=0)
    visualize_and_save_matrix(average_matrix, "average", fisher, paradigm, output_folder)