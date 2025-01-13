import os
import sys

# Root paths
TEST_ROOT = os.path.dirname(__file__)  # Root of test folder
PROJECT_ROOT = os.path.dirname(TEST_ROOT)  # Root of project
PATH_DATA = os.path.join(PROJECT_ROOT, "data")  # Root of data

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(PROJECT_ROOT, 'src')))
