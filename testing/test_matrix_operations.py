import pytest
import numpy as np
from data_processing.matrix_operations import fisher_transform, calculate_region_averages

def test_fisher_transform():
    """Test Fisher transformation."""
    corr_matrix = np.array([[0.5, -0.8], [0.2, 0.9]])
    transformed = fisher_transform(corr_matrix)

    assert transformed.shape == corr_matrix.shape, "Output shape mismatch"
    assert np.all(transformed < 5), "Unexpected large values after transformation"

def test_calculate_region_averages():
    """Test region-wise averaging of matrices."""
    test_matrix = np.random.rand(30, 30)
    lang_avg, md_avg, lang_md_avg = calculate_region_averages(test_matrix)

    assert isinstance(lang_avg, float), "Language average should be a float"
    assert isinstance(md_avg, float), "MD average should be a float"
    assert isinstance(lang_md_avg, float), "Lang-MD average should be a float"
