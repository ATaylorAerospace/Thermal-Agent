"""
Pytest configuration — ensures the project root is on sys.path.

Author: A Taylor
"""

import sys
from pathlib import Path

# Add project root to sys.path so `from src.xxx import ...` works everywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
