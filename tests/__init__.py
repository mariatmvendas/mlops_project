"""
This script sets up the root paths for the testing environment and ensures 
that the project's `src` directory is accessible by modifying `sys.path`.

- TEST_ROOT: The root directory of the test folder.
- PROJECT_ROOT: The root directory of the project, derived from the test folder's location.
- PATH_DATA: The root directory where data is stored.

Additionally, the script appends the `src` directory to `sys.path` to enable 
importing modules from the source code directory during testing.
"""

import os
import sys

# Root paths
TEST_ROOT = os.path.dirname(__file__)  # Root of test folder
PROJECT_ROOT = os.path.dirname(TEST_ROOT)  # Root of project
PATH_DATA = os.path.join(PROJECT_ROOT, "data")  # Root of data

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(PROJECT_ROOT, 'src')))
