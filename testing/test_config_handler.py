import os
import pytest
from config.config_handler import create_config, read_config

def test_create_config():
    """Test that config file is created successfully."""
    create_config()
    assert os.path.exists("config.ini"), "Config file was not created!"

def test_read_config():
    """Test that config values are correctly retrieved."""
    create_config()  # Ensure file exists
    config_values = read_config()
    
    assert isinstance(config_values, dict), "Config should return a dictionary"
    assert "fisher" in config_values, "Missing 'fisher' key in config"
    assert "paradigm" in config_values, "Missing 'paradigm' key in config"
    assert isinstance(config_values["language_list"], list), "'language_list' should be a list"