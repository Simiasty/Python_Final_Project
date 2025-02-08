
# Project Name: Brain Network Correlation Analysis

## Authors: Jimena Mendez, Samuel Hapeta

## Project Description
This project analyzes the correlation between different brain networks during distinct cognitive paradigms. The analysis focuses on computing and visualizing correlation matrices for different languages and extracting meaningful insights using statistical methods.

Our work aims to replicate the methodology decribed in parts of the selected article (see article references below).

We have replicated the visualizations included in figures 7 and 3c from the original article. 
We have also added our own element of analysis and visualization in figure 11 (see included pdf file for more details).

### Objectives:
- Compute correlation matrices for brain regions involved in language processing and multiple-demand (MD) networks.
- Apply Fisher transformation when enabled.
- Generate visualizations, including boxplots and integration vs. dissociation plots, to interpret the results.
- Structure the project for easy reproducibility and further modifications.

### Assumptions & Hypothesis:
- Brain network activity varies between different cognitive paradigms (e.g., Story Comprehension vs. Resting State).
- Correlation between language-related brain regions differs across languages.
- The application of the Fisher transformation enhances interpretability.

---
## Folder and Module Structure
```
├── config/                     # Configuration file handler
│   ├── config_handler.py       # Reads and writes config settings
├── data_processing/            # Handles data extraction and processing
│   ├── file_processing.py      # Reads and processes raw input files
│   ├── matrix_operations.py    # Applies transformations to correlation matrices
├── visualization/              # Visualization utilities
│   ├── matrix_plot.py          # Plots and saves correlation matrices
│   ├── boxplot.py              # Generates boxplots
│   ├── integration_plot.py     # Creates integration vs. dissociation plots
├── tests/                      # Unit tests for different components
│   ├── test_file_processing.py
│   ├── test_matrix_operations.py
│   ├── test_visualization.py
│   ├── test_config_handler.py
├── Data/                       # Data directory (should contain CSV files)
├── Main.py                     # Main script to run the analysis
├── README.md                   # Project documentation
```

---
## Key Stages of the Project
1. **Data Import**: Loads time-series data from CSV files.
2. **Data Processing**: Computes region-wise correlation matrices for different languages.
3. **Statistical Transformations**: Applies Fisher transformation if enabled.
4. **Analysis**:
   - Compute inter- and intra-network correlation averages.
   - Extract meaningful statistical components for visualization.
5. **Visualization**:
   - Correlation matrices
   - Boxplots showing statistical distributions
   - Integration vs. Dissociation analysis

---
## Data Description & Link to Dataset
The dataset consists of time-series data from brain activity recordings during different cognitive tasks. Each file corresponds to a different participant and contains numerical values representing recorded activity averaged across regions of interest. The dataset should be placed under the `Data/` directory.

Link to the entire dataset: https://osf.io/cw89s/

For the purposes of this project, the files included in the `Data/` directory of this repository were used.

---
## References
- (Malik-Moraleda, S., Ayyash, D., Gallée, J. et al. An investigation across 45 languages and 12 language families reveals a universal language network. Nat Neurosci 25, 1014–1019 (2022). https://doi.org/10.1038/s41593-022-01114-5)

---
## Instructions for Running the Project
1. **Install dependencies**:
   ```bash
   pip install -e .
   ```
2. **Configure settings**:
   - Edit `config/config_handler.py` or generate a config file.
3. **Ensure data is placed in the `Data/` folder.**
4. **Run the analysis**:
   ```bash
   python Main.py
   ```
5. **Results**:
   - Generated visualizations will be saved in the output directory specified in the config.

---
### Testing
To run unit tests:
```bash
pytest tests/
```

