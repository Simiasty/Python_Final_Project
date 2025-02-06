import pytest
import numpy as np
import os
import matplotlib.pyplot as plt

from visualization.boxplot import calculate_boxplot_components, plot_custom_boxplot
from visualization.integration_plot import plot_integration_vs_dissociation
from visualization.matrix_plot import visualize_and_save_matrix

def test_calculate_boxplot_components():
    """Ensure correct computation of boxplot statistics."""
    data = {"Language_avg": [0.1, 0.2, 0.3, 0.4, 0.5]}
    components = calculate_boxplot_components(data)

    assert "Language_avg" in components, "Missing category in output"
    assert "median" in components["Language_avg"], "Missing median value"

def test_plot_custom_boxplot(tmp_path):
    """Test that boxplot function does not crash and generates output."""
    data = {"Language_avg": [0.1, 0.2, 0.3, 0.4, 0.5]}
    components = calculate_boxplot_components(data)
    output_folder = tmp_path / "output"
    output_folder.mkdir()

    try:
        plot_custom_boxplot(components, ["Language_avg"], data, fisher=True, paradigm="story", output_folder=str(output_folder))
        assert (output_folder / "boxplot.png").exists(), "Plot file was not created"
    except Exception as e:
        pytest.fail(f"Boxplot function raised an error: {e}")

def test_plot_integration_vs_dissociation(tmp_path):
    """Ensure integration vs. dissociation plot runs without error."""
    region_averages = {"English": {"Integration_Language": 0.3, "Integration_MD": 0.2, "Dissociation": 0.1}}
    output_folder = tmp_path / "output"
    output_folder.mkdir()

    try:
        plot_integration_vs_dissociation(region_averages, fisher=True, paradigm="story", output_folder=str(output_folder))
        assert (output_folder / "Fig11.png").exists(), "Plot file was not created"
    except Exception as e:
        pytest.fail(f"Integration plot function raised an error: {e}")

def test_visualize_and_save_matrix(tmp_path):
    """Ensure matrix visualization function runs without error."""
    matrix = np.random.rand(30, 30)
    output_folder = tmp_path / "output"
    output_folder.mkdir()

    try:
        visualize_and_save_matrix(matrix, "Test_Language", fisher=True, paradigm="story", output_folder=str(output_folder))
        assert (output_folder / "Test_Language.png").exists(), "Matrix plot file was not created"
    except Exception as e:
        pytest.fail(f"Matrix plot function raised an error: {e}")