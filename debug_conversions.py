#!/usr/bin/env python3
"""Debug script to figure out the conversion discrepancy."""

import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))
from scripts.fusion.quick_analysis import calculate_always_override_conversions

# Run the conversion calculation
base_run = "53vig20u" 
rerun_id = "9qup1u07"

print("Calculating conversions...")
conversions = calculate_always_override_conversions(base_run, rerun_id)
print(f"Final conversions: {conversions}")
