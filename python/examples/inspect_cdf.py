#!/usr/bin/env python3
"""
Example: Inspect CDF file metadata
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdf import ConscientDataFormat

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_cdf.py <cdf_file>")
        sys.exit(1)
    
    cdf_file = Path(sys.argv[1])
    if not cdf_file.exists():
        print(f"Error: File not found: {cdf_file}")
        sys.exit(1)
    
    cdf = ConscientDataFormat()
    info = cdf.get_cdf_info(cdf_file)
    
    print(info)

if __name__ == "__main__":
    main()

