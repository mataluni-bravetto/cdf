#!/usr/bin/env python3
"""
Example: Encode a file to CDF format
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdf import ConscientDataFormat

def main():
    if len(sys.argv) < 2:
        print("Usage: python encode_file.py <input_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Encoding: {input_file}")
    
    cdf = ConscientDataFormat()
    output_file = cdf.save_cdf(input_file)
    
    print(f"✅ Created: {output_file}")
    
    # Show info
    info = cdf.get_cdf_info(output_file)
    print()
    print(info)

if __name__ == "__main__":
    main()

