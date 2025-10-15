import pytest

def test_example():
    """A simple test to verify pytest works"""
    assert 1 + 1 == 2

def test_imports():
    """Test that required libraries can be imported"""
    try:
        import pandas as pd
        import numpy as np
        assert True
    except ImportError:
        pytest.fail("Required libraries not installed")
