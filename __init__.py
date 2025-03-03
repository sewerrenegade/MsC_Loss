import sys
import os

# Get the directory of this file (which is mypackage/)
package_dir = os.path.dirname(os.path.abspath(__file__))

# Add this directory to sys.path (if not already added)
if package_dir not in sys.path:
    sys.path.append(package_dir)