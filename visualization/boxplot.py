import matplotlib.pyplot as plt
import os

def calculate_boxplot_components(data):
    """
    Calculates components for the boxplot.
    
    Input:
        - data (dict): Dictionary containing numerical values for different categories.
        
    Output:
        - components (dict): Dictionary with calculated statistics (q1, q3, median, IQR).
    """
    components = {}
    for category, values in data.items():
        values = sorted(values)
        q1 = values[int(0.25 * len(values))]
        q3 = values[int(0.75 * len(values))]
        median = values[int(0.5 * len(values))]
        iqr = q3 - q1  # Interquartile range
        components[category] = {
            "q1": q1,
            "q3": q3,
            "median": median,
            "iqr": iqr
        }
    return components

def plot_custom_boxplot(components, category_labels, data, fisher, paradigm, output_folder):
    """
    Plots a custom boxplot.

    Inputs:
        - components (dict): Statistical components of the boxplot.
        - category_labels (list): Labels for the x-axis.
        - data (dict): Raw data points for each category.
        - fisher (bool): Determines if Fisher transform is applied.
        - paradigm (str): Experiment paradigm used.
        - output_folder (str): Directory to save the plot.
    """
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, category in enumerate(category_labels):
        points = data[category]
        ax.scatter([i] * len(points), points, color='black', alpha=0.2, s=15, zorder=1)

    ax.set_xticks(range(len(category_labels)))
    ax.set_xticklabels(category_labels)
    ax.set_ylabel('Average correlation')
    plt.savefig(os.path.join(output_folder, "boxplot.png"))
    plt.show()

