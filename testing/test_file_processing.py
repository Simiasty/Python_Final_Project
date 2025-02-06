import pytest
import numpy as np
import os
from data_processing.file_processing import process_files_in_folder, process_language_files

def test_process_files_in_folder(tmp_path):
    """Test processing of CSV files in a folder."""
    # Create temp CSV file
    test_folder = tmp_path / "test_data"
    test_folder.mkdir()
    test_file = test_folder / "English_test.csv"
    test_file.write_text("col1,col2,col3,col4,col5\n1,2,3,4,5\n6,7,8,9,10")

    result = process_files_in_folder(str(test_folder), "English")
    
    assert isinstance(result, list), "Output should be a list"
    assert len(result) == 1, "Should have processed one file"
    assert result[0].shape == (1, 1), "Should extract only the last column's values"

def test_process_language_files(tmp_path):
    """Test processing of language and MD matrices."""
    lang_folder = tmp_path / "lang_data"
    md_folder = tmp_path / "md_data"
    lang_folder.mkdir()
    md_folder.mkdir()

    # Create mock CSV files
    for folder, name in [(lang_folder, "English_lang.csv"), (md_folder, "English_md.csv")]:
        (folder / name).write_text("col1,col2,col3,col4,col5\n1,2,3,4,5\n6,7,8,9,10")

    lang_matrix, md_matrix = process_language_files("English", str(lang_folder), str(md_folder))

    assert isinstance(lang_matrix, np.ndarray), "Language matrix should be a NumPy array"
    assert isinstance(md_matrix, np.ndarray), "MD matrix should be a NumPy array"
    assert lang_matrix.shape == (1,), "Unexpected shape for language matrix"
    assert md_matrix.shape == (1,), "Unexpected shape for MD matrix"