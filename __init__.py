import sys
import os

# Get the directory of this file (which is mypackage/)
package_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(package_dir)
# Add this directory to sys.path (if not already added)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"added this path:{parent_dir}")
