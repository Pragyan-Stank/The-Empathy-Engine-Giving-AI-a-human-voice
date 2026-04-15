import sys
import os

# Add the project root to sys.path so app modules can be imported in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
