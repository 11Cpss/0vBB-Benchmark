#!/usr/bin/env python3
"""Render the separate SuperNEMO energy-only illustration from repository CSVs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from illustration import plot_main
if __name__=='__main__':plot_main()
