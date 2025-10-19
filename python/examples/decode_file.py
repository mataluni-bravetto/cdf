#!/usr/bin/env python3
"""
Example: Decode a CDF file back to original
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdf import ConscientDataFormat

def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_file.py <cdf_file>")
        sys.exit(1)
    
    cdf_file = Path(sys.argv[1])
    if not cdf_file.exists():
        print(f"Error: File not found: {cdf_file}")
        sys.exit(1)
    
    print(f"Decoding: {cdf_file}")
    
    cdf = ConscientDataFormat()
    output_file = cdf.extract_cdf(cdf_file)
    
    print(f"✅ Extracted: {output_file}")

if __name__ == "__main__":
    main()

