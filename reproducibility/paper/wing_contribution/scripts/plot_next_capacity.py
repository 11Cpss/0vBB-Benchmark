#!/usr/bin/env python3
"""Render the NEXT figure using the reviewed CSV and results stored in this repository."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from plot_next import main
if __name__=='__main__':main()
