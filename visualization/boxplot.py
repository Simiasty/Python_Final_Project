import matplotlib.pyplot as plt
import os
import numpy as np

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

    # Validate components - Added for debugging
    if not isinstance(components, dict):
        raise TypeError(f"Input 'components' must be a dictionary. Got: {type(components)}")
    for key in category_labels:
        if key not in components:
            raise KeyError(f"Key '{key}' not found in 'components'. Expected keys: {category_labels}") 

    # Validate output folder - Added for debugging
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    print(f"Plotting boxplot for categories: {category_labels}")

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
        if (fisher):
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

