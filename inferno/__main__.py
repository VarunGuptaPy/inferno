"""
Main entry point for the Inferno package.
This allows running the package directly with `python -m inferno`.
"""

import sys
from inferno.cli import main

if __name__ == "__main__":
    sys.exit(main())
